import asyncio
import logging as py_logging

from amaranth import *
from amaranth.lib import io
from amaranth.sim import Simulator

from glasgow.gateware.ports import PortGroup
from glasgow.applet import GlasgowAppletV2TestCase, synthesis_test

from . import (DebugHCS08Applet, DebugHCS08Component, DebugHCS08Interface, HCS08Error, BDCStatus,
               _Command, _BDC, _BIT_CYCLES, _RX_SAMPLE, _SYNC_RESPONSE, _DVF_RETRIES,
               _TX_ONE_RELEASE, _TX_ZERO_HOLD, parse_srecord, coalesce)


class _StubPipe:
    """A pipe that records each flushed request and replays a scripted response for it."""

    def __init__(self, responses):
        self._responses = list(responses)
        self._pending = bytearray()
        self._current = b""
        self.sent = []

    async def send(self, data):
        self._pending += data

    async def flush(self):
        self.sent.append(bytes(self._pending))
        self._pending = bytearray()
        if not self._responses:
            raise AssertionError(f"unexpected request #{len(self.sent)}: {self.sent[-1].hex()}")
        self._current = self._responses.pop(0)

    async def recv(self, length):
        assert length == len(self._current), \
            f"expected to read {len(self._current)} bytes, asked for {length}"
        return self._current


DIVISOR = 4 # sys clock cycles per simulated target BDC clock cycle


class _Harness(Elaboratable):
    """The BDC component wired to a pseudo-open-drain BKGD net with a target model attached."""

    def __init__(self):
        self.bkgd = io.SimulationPort("io", 1)
        self.target_low = Signal() # when high, the modelled target pulls BKGD low
        # A short sys clock period keeps the SYNC request (fixed at 1 ms) cheap to simulate.
        self.dut = DebugHCS08Component(
            PortGroup(bkgd=self.bkgd, reset=None), sys_clk_period=1e-5)

    def elaborate(self, platform):
        m = Module()
        m.submodules.dut = self.dut
        # Both ends are open-drain with a pullup: the net is low if either side pulls it low.
        host_low = self.bkgd.oe[0] & ~self.bkgd.o[0]
        m.d.comb += self.bkgd.i.eq(~(host_low | self.target_low))
        return m


async def _send(ctx, dut, data):
    ctx.set(dut.i_stream.valid, 1)
    for byte in data:
        ctx.set(dut.i_stream.payload, byte)
        while not ctx.get(dut.i_stream.ready):
            await ctx.tick()
        await ctx.tick() # complete the transfer
    ctx.set(dut.i_stream.valid, 0)


async def _recv(ctx, dut, count):
    result = []
    ctx.set(dut.o_stream.ready, 1)
    for _ in range(count):
        while not ctx.get(dut.o_stream.valid):
            await ctx.tick()
        # Sample the payload while it is still being offered; ticking first would read back
        # whatever the component drives in the state it moves to once the transfer completes.
        result.append(ctx.get(dut.o_stream.payload))
        await ctx.tick()
    ctx.set(dut.o_stream.ready, 0)
    return result


# Once `_send` returns, the component has just entered the state that executes the command, so
# the next simulated cycle is cycle 0 of that command. Host-side bit timing is fully determined
# by the divisor from there on, so the testbenches below count cycles rather than chase edges:
# a receiving target has no edges of its own to synchronise against anyway.


class DebugHCS08AppletTestCase(GlasgowAppletV2TestCase, applet=DebugHCS08Applet):
    @synthesis_test
    def test_build(self):
        self.assertBuilds()

    def _simulate(self, testbench):
        harness = _Harness()
        sim = Simulator(harness)
        sim.add_clock(1e-5)

        async def wrapper(ctx):
            await testbench(ctx, harness)

        sim.add_testbench(wrapper)
        sim.run()

    def test_transmit_bit_timing(self):
        """The target must see each transmitted bit at its sampling point, 10 cycles in."""
        trace = []

        async def testbench(ctx, harness):
            dut = harness.dut
            await _send(ctx, dut, [_Command.SetDivisor.value, DIVISOR, 0x00,
                                   _Command.Transmit.value, 0xA5])
            for _ in range(8 * _BIT_CYCLES * DIVISOR):
                trace.append(ctx.get(harness.bkgd.i))
                await ctx.tick()

        self._simulate(testbench)

        bits, low_cycles = [], []
        for index in range(8):
            start = index * _BIT_CYCLES * DIVISOR
            # What the target sees when it samples, about 10 BDC cycles into the bit time.
            bits.append(trace[start + _RX_SAMPLE * DIVISOR])
            # How long the host held the net low, in whole BDC cycles.
            held = [trace[start + cycle * DIVISOR] for cycle in range(_BIT_CYCLES)]
            low_cycles.append(held.index(1) if 1 in held else _BIT_CYCLES)

        self.assertEqual(bits, [1, 0, 1, 0, 0, 1, 0, 1]) # 0xA5, MSB first
        # A one is a short low pulse and a zero a long one, both straddling cycle 10.
        self.assertEqual(low_cycles, [_TX_ONE_RELEASE if bit else _TX_ZERO_HOLD for bit in bits])

    def test_receive_bit_timing(self):
        """A target driving per HCS08RMV1 Figure 7-3/7-4 must be received correctly."""
        received = []

        async def testbench(ctx, harness):
            dut = harness.dut
            await _send(ctx, dut, [_Command.SetDivisor.value, DIVISOR, 0x00,
                                   _Command.Receive.value])

            async def drive_bit(bit):
                if bit:
                    # Logic 1: the target drives only a brief speedup pulse at cycle 7, which on
                    # a pulled-up net is indistinguishable from leaving it released.
                    for _ in range(_BIT_CYCLES * DIVISOR):
                        await ctx.tick()
                else:
                    # Logic 0: the target holds the net low for 13 cycles, then releases.
                    ctx.set(harness.target_low, 1)
                    for _ in range(13 * DIVISOR):
                        await ctx.tick()
                    ctx.set(harness.target_low, 0)
                    for _ in range(3 * DIVISOR):
                        await ctx.tick()

            for bit in [0, 1, 0, 1, 1, 0, 1, 0]: # 0x5A, MSB first
                await drive_bit(bit)
            received.extend(await _recv(ctx, dut, 1))

        self._simulate(testbench)
        self.assertEqual(received, [0x5A])

    def test_sync_measurement(self):
        """SYNC must measure the length of the target's 128-cycle response pulse."""
        measured = []

        async def testbench(ctx, harness):
            dut = harness.dut
            await _send(ctx, dut, [_Command.Sync.value])
            # Wait out the host's own low pulse and its speedup pulse, then let the net settle.
            while not ctx.get(harness.bkgd.i):
                await ctx.tick()
            # The target waits for BKGD to return high and delays 16 cycles before answering.
            for _ in range(_BIT_CYCLES * DIVISOR):
                await ctx.tick()
            # Respond with a 128 BDC-cycle low pulse.
            ctx.set(harness.target_low, 1)
            for _ in range(_SYNC_RESPONSE * DIVISOR):
                await ctx.tick()
            ctx.set(harness.target_low, 0)
            low, high = await _recv(ctx, dut, 2)
            measured.append(low | (high << 8))

        self._simulate(testbench)
        count, = measured
        # The measurement is bounded by the input synchroniser latency at either end.
        self.assertAlmostEqual(count, _SYNC_RESPONSE * DIVISOR, delta=4)
        self.assertEqual(max(1, round(count / _SYNC_RESPONSE)), DIVISOR)

    def test_sync_no_target(self):
        """SYNC must report 0 rather than hang when nothing answers."""
        measured = []

        async def testbench(ctx, harness):
            dut = harness.dut
            await _send(ctx, dut, [_Command.Sync.value])
            low, high = await _recv(ctx, dut, 2)
            measured.append(low | (high << 8))

        self._simulate(testbench)
        self.assertEqual(measured, [0])

    def test_flash_divider(self):
        """FCDIV must be chosen so the FLASH clock lands within 150..200 kHz."""
        for bus_frequency in (1e6, 2e6, 4e6, 8e6, 10e6, 20e6):
            prdiv8, div = DebugHCS08Interface._flash_divider(bus_frequency)
            fclk = bus_frequency / ((8 if prdiv8 else 1) * (div + 1))
            self.assertTrue(150e3 <= fclk <= 200e3,
                            f"bus={bus_frequency} gives fclk={fclk}")
        # Table 4-7 of MC9S08AW60 gives these exact settings.
        self.assertEqual(DebugHCS08Interface._flash_divider(10e6), (False, 49))
        self.assertEqual(DebugHCS08Interface._flash_divider(8e6), (False, 39))
        self.assertEqual(DebugHCS08Interface._flash_divider(4e6), (False, 19))
        # Beyond bus/(8*64) the slowest available divider still overshoots 200 kHz.
        with self.assertRaises(HCS08Error):
            DebugHCS08Interface._flash_divider(200e6)

    def test_parse_srecord(self):
        # S1 record: length 0x07, address 0x1860, data 01 02 03 04, checksum.
        memory = parse_srecord("S10718600102030476\nS9030000FC\n")
        self.assertEqual(memory, {0x1860: 1, 0x1861: 2, 0x1862: 3, 0x1863: 4})

    def test_parse_srecord_rejects_bad_checksum(self):
        with self.assertRaises(HCS08Error):
            parse_srecord("S10718600102030477\n")

    def test_parse_srecord_rejects_empty(self):
        with self.assertRaises(HCS08Error):
            parse_srecord("S9030000FC\n")

    # === DVF recovery ===
    #
    # DVF only arises when the BDC's bus-cycle steal loses a race with a running target CPU, which
    # cannot be provoked on demand, so the recovery is exercised against a scripted pipe instead.

    def _stub_interface(self, responses):
        """An interface wired to a fake pipe replaying `responses`, recording what was sent."""
        iface = object.__new__(DebugHCS08Interface)
        iface._logger = py_logging.getLogger(__name__)
        iface._level = py_logging.DEBUG
        iface._divisor = 6
        iface._pipe = _StubPipe(responses)
        return iface

    def test_dvf_read_is_reissued(self):
        """A read reporting DVF is retried until it succeeds, and returns the retried data."""
        iface = self._stub_interface([
            bytes([BDCStatus.DVF, 0x00]),   # first attempt: access did not happen
            bytes([BDCStatus.DVF, 0x00]),   # second attempt: still lost the race
            bytes([0x00, 0xA5]),            # third attempt: succeeded
        ])
        value = asyncio.run(iface.read(0x1234, 1))
        self.assertEqual(value, b"\xa5")
        # Every attempt must re-send the original address, since READ_LAST cannot be used here.
        self.assertEqual(len(iface._pipe.sent), 3)
        for request in iface._pipe.sent:
            self.assertEqual(request[1], _BDC.READ_BYTE_WS)
            self.assertEqual(request[3], 0x12)
            self.assertEqual(request[5], 0x34)

    def test_dvf_write_is_not_reissued(self):
        """A write reporting DVF is waited out with READ_STATUS, never repeated.

        Reissuing would program a FLASH byte twice, which is forbidden without an erase.
        """
        iface = self._stub_interface([
            bytes([BDCStatus.DVF]),  # the write itself: latched, but not yet complete
            bytes([BDCStatus.DVF]),  # READ_STATUS: still outstanding
            bytes([0x00]),           # READ_STATUS: latched write has completed
        ])
        asyncio.run(iface.write(0x1234, b"\xa5"))
        opcodes = [request[1] for request in iface._pipe.sent]
        self.assertEqual(opcodes, [_BDC.WRITE_BYTE_WS, _BDC.READ_STATUS, _BDC.READ_STATUS])
        self.assertEqual(sum(1 for op in opcodes if op == _BDC.WRITE_BYTE_WS), 1)

    def test_dvf_read_gives_up(self):
        """Persistent DVF raises rather than looping forever."""
        iface = self._stub_interface([bytes([BDCStatus.DVF, 0x00])] * (_DVF_RETRIES + 1))
        with self.assertRaisesRegex(HCS08Error, "DVF"):
            asyncio.run(iface.read(0x1234, 1))

    def test_batch_retries_only_failed_accesses(self):
        """Only the accesses that reported DVF are reissued, not the whole batch."""
        first = bytes([0x00, 0x11]) + bytes([BDCStatus.DVF, 0x00]) + bytes([0x00, 0x33])
        iface = self._stub_interface([first, bytes([0x00, 0x22])])
        self.assertEqual(asyncio.run(iface.read(0x2000, 3)), b"\x11\x22\x33")
        # The retry carries exactly one command, for the middle address only: opcode, two address
        # bytes and a delay, each Transmit-prefixed, then two Receive opcodes.
        retry = iface._pipe.sent[1]
        self.assertEqual(len(retry), 9)
        self.assertEqual((retry[1], retry[3], retry[5]), (_BDC.READ_BYTE_WS, 0x20, 0x01))

    def test_coalesce(self):
        self.assertEqual(
            coalesce({0x10: 0xAA, 0x11: 0xBB, 0x20: 0xCC}),
            [(0x10, b"\xaa\xbb"), (0x20, b"\xcc")])
