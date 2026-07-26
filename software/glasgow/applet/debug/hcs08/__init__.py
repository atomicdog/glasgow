# Ref: HCS08 Family Reference Manual, Rev. 2 (HCS08RMV1) §7.3 Background Debug Controller (BDC)
# Document Number: HCS08RMV1
# Ref: MC9S08AW60 Data Sheet, Rev 2 §4 Memory (FLASH command interface)
# Document Number: MC9S08AW60
# Ref: AN3335, Introduction to HCS08 Background Debug Mode
# Document Number: AN3335

# The BDC is a single-wire debug interface on the BKGD pin. Unlike most debug interfaces there is
# no clock line: bit timing is referenced to the *target's* BDC clock, whose frequency the host
# does not know a priori. The host recovers it with the SYNC command, which asks the target to
# emit a 128 BDC-cycle low pulse that the host measures.
#
# Every bit time is started by a host-driven falling edge and lasts 16 target BDC cycles. Within
# a bit time the target samples (host-to-target) or drives (target-to-host) around cycle 10, so
# the gateware works in units of target BDC cycles throughout, with `divisor` giving the number of
# sys clock cycles per target BDC cycle.

import enum as py_enum
import struct
import asyncio
import argparse

from amaranth import *
from amaranth.lib import io, cdc, enum, wiring, stream
from amaranth.lib.wiring import In, Out

from glasgow.support import logging
from glasgow.abstract import AbstractAssembly, GlasgowPin
from glasgow.applet import GlasgowAppletV2, GlasgowAppletError


__all__ = ["DebugHCS08Interface", "HCS08Error", "BDCStatus"]


class _Command(enum.Enum, shape=8):
    """Opcodes of the command stream consumed by :class:`DebugHCS08Component`."""

    SetDivisor = 0x00 # + u16le: sys clock cycles per target BDC clock cycle
    Sync       = 0x01 # -> u16le: measured low time of the sync response, in sys clock cycles
    Transmit   = 0x02 # + u8: byte to shift out, MSB first
    Receive    = 0x03 # -> u8: byte shifted in, MSB first
    Delay      = 0x04 # delay 16 target BDC cycles (the `d` element of a BDC command)
    SetReset   = 0x05 # + u8: 0 drives RESET low, 1 releases it
    SetBkgd    = 0x06 # + u8: 0 drives BKGD low, 1 drives it high, 2 releases it


# Bit times, in target BDC clock cycles. The target samples a host-driven bit at cycle 10 and
# drives a target-to-host bit until cycle 13 (logic 0) or from cycle 7 (logic 1 speedup pulse).
_BIT_CYCLES     = 16
_TX_ONE_RELEASE = 4  # drive low for cycles 0..3, then high: target sees 1 at cycle 10
_TX_ZERO_HOLD   = 13 # drive low for cycles 0..12, then high: target sees 0 at cycle 10
_RX_DRIVE       = 3  # host must hold low >=2 cycles, then release before the target drives
_RX_SAMPLE      = 10 # host samples about 10 cycles after it started the bit time
_SYNC_RESPONSE  = 128 # the target's sync response pulse is 128 BDC cycles long


class DebugHCS08Component(wiring.Component):
    i_stream: In(stream.Signature(8))
    o_stream: Out(stream.Signature(8))

    def __init__(self, ports, *, sys_clk_period):
        self._ports = ports
        # The SYNC request must be low for at least 128 cycles of the slowest BDC clock we could
        # ever face. 1 ms covers a target BDC clock down to 128 kHz, well below any real part.
        self._sync_low_cyc = int(1e-3 / sys_clk_period)
        # Give the target 4x the request length to begin its response before declaring failure.
        self._sync_wait_cyc = self._sync_low_cyc * 4

        super().__init__()

    def elaborate(self, platform):
        m = Module()

        m.submodules.bkgd_buffer = bkgd_buffer = io.Buffer("io", self._ports.bkgd)
        bkgd_i = Signal(init=1)
        m.submodules += cdc.FFSynchronizer(bkgd_buffer.i, bkgd_i, init=1)

        if self._ports.reset is not None:
            m.submodules.reset_buffer = reset_buffer = io.Buffer("io", self._ports.reset)
            # RESET is an open-drain signal: drive low, or release and let the target pull up.
            reset_assert = Signal()
            m.d.comb += reset_buffer.o.eq(0)
            m.d.comb += reset_buffer.oe.eq(reset_assert)
        else:
            reset_assert = Signal()

        divisor = Signal(16, init=1)

        # Level driven onto BKGD whenever no transfer is in progress. Released by default, so
        # the target's on-chip pullup defines the idle level, but the host can hold BKGD low
        # across a reset to select active background mode.
        idle_o  = Signal()
        idle_oe = Signal()
        m.d.comb += bkgd_buffer.o.eq(idle_o)
        m.d.comb += bkgd_buffer.oe.eq(idle_oe)

        # `tick` pulses once per target BDC clock cycle.
        tick_ctr = Signal(16)
        tick     = Signal()
        tick_rst = Signal()
        m.d.comb += tick.eq(tick_ctr == 0)
        with m.If(tick_rst | tick):
            m.d.sync += tick_ctr.eq(divisor - 1)
        with m.Else():
            m.d.sync += tick_ctr.eq(tick_ctr - 1)

        bit_ctr   = Signal(range(_BIT_CYCLES)) # position within the current bit time
        bit_index = Signal(range(8))
        shreg     = Signal(8)
        is_rx     = Signal()

        timer   = Signal(range(max(self._sync_wait_cyc, self._sync_low_cyc) + 1))
        measure = Signal(16)

        # Drive levels during a bit time. Outside of shifting the pin is released and held high by
        # the target's on-chip pullup.
        tx_bit    = shreg[7]
        tx_thresh = Mux(tx_bit, _TX_ONE_RELEASE, _TX_ZERO_HOLD)

        with m.FSM():
            with m.State("Read Command"):
                m.d.comb += self.i_stream.ready.eq(1)
                # Keep the bit clock primed so the first bit time is full length.
                m.d.comb += tick_rst.eq(1)
                with m.If(self.i_stream.valid):
                    m.d.sync += bit_ctr.eq(0)
                    m.d.sync += bit_index.eq(0)
                    with m.Switch(self.i_stream.payload):
                        with m.Case(_Command.SetDivisor):
                            m.next = "Divisor Low"
                        with m.Case(_Command.Sync):
                            m.d.sync += timer.eq(self._sync_low_cyc - 1)
                            m.next = "Sync Low"
                        with m.Case(_Command.Transmit):
                            m.next = "Transmit Data"
                        with m.Case(_Command.Receive):
                            m.d.sync += is_rx.eq(1)
                            m.next = "Shift"
                        with m.Case(_Command.Delay):
                            m.next = "Delay"
                        with m.Case(_Command.SetReset):
                            m.next = "Reset Data"
                        with m.Case(_Command.SetBkgd):
                            m.next = "Bkgd Data"

            with m.State("Divisor Low"):
                m.d.comb += self.i_stream.ready.eq(1)
                with m.If(self.i_stream.valid):
                    m.d.sync += divisor[0:8].eq(self.i_stream.payload)
                    m.next = "Divisor High"

            with m.State("Divisor High"):
                m.d.comb += self.i_stream.ready.eq(1)
                with m.If(self.i_stream.valid):
                    m.d.sync += divisor[8:16].eq(self.i_stream.payload)
                    m.next = "Read Command"

            with m.State("Reset Data"):
                m.d.comb += self.i_stream.ready.eq(1)
                with m.If(self.i_stream.valid):
                    m.d.sync += reset_assert.eq(self.i_stream.payload == 0)
                    m.next = "Read Command"

            with m.State("Bkgd Data"):
                m.d.comb += self.i_stream.ready.eq(1)
                with m.If(self.i_stream.valid):
                    m.d.sync += idle_o.eq(self.i_stream.payload == 1)
                    m.d.sync += idle_oe.eq(self.i_stream.payload != 2)
                    m.next = "Read Command"

            with m.State("Transmit Data"):
                m.d.comb += self.i_stream.ready.eq(1)
                with m.If(self.i_stream.valid):
                    m.d.sync += shreg.eq(self.i_stream.payload)
                    m.d.sync += is_rx.eq(0)
                    m.next = "Shift"

            # One state shifts a whole byte, in either direction, MSB first.
            with m.State("Shift"):
                with m.If(is_rx):
                    m.d.comb += bkgd_buffer.o.eq(0)
                    m.d.comb += bkgd_buffer.oe.eq(bit_ctr < _RX_DRIVE)
                with m.Else():
                    m.d.comb += bkgd_buffer.o.eq(bit_ctr >= tx_thresh)
                    m.d.comb += bkgd_buffer.oe.eq(1)

                with m.If(tick):
                    with m.If(is_rx & (bit_ctr == _RX_SAMPLE)):
                        m.d.sync += shreg.eq(Cat(bkgd_i, shreg[0:7]))
                    with m.If(bit_ctr == _BIT_CYCLES - 1):
                        m.d.sync += bit_ctr.eq(0)
                        with m.If(~is_rx):
                            m.d.sync += shreg.eq(Cat(C(0, 1), shreg[0:7]))
                        m.d.sync += bit_index.eq(bit_index + 1)
                        with m.If(bit_index == 7):
                            with m.If(is_rx):
                                m.next = "Receive Data"
                            with m.Else():
                                m.next = "Read Command"
                    with m.Else():
                        m.d.sync += bit_ctr.eq(bit_ctr + 1)

            with m.State("Receive Data"):
                m.d.comb += self.o_stream.payload.eq(shreg)
                m.d.comb += self.o_stream.valid.eq(1)
                with m.If(self.o_stream.ready):
                    m.next = "Read Command"

            # A `d` element: 16 BDC cycles of idle to let the target complete a CPU operation.
            with m.State("Delay"):
                with m.If(tick):
                    with m.If(bit_ctr == _BIT_CYCLES - 1):
                        m.next = "Read Command"
                    with m.Else():
                        m.d.sync += bit_ctr.eq(bit_ctr + 1)

            # SYNC: drive a long low, release, then measure the target's 128-cycle response.
            with m.State("Sync Low"):
                m.d.comb += bkgd_buffer.o.eq(0)
                m.d.comb += bkgd_buffer.oe.eq(1)
                m.d.sync += timer.eq(timer - 1)
                with m.If(timer == 0):
                    m.next = "Sync Speedup"

            with m.State("Sync Speedup"):
                # One cycle of active high to force a fast rise before going high-impedance.
                m.d.comb += bkgd_buffer.o.eq(1)
                m.d.comb += bkgd_buffer.oe.eq(1)
                m.d.sync += timer.eq(self._sync_wait_cyc - 1)
                m.d.sync += measure.eq(0)
                m.next = "Sync Release"

            with m.State("Sync Release"):
                # The input synchroniser still reports our own low drive for a couple of cycles
                # after the pin is released. Wait for it to catch up with the released line,
                # otherwise the stale low reads as a zero-length response from the target.
                m.d.sync += timer.eq(timer - 1)
                with m.If(bkgd_i):
                    m.next = "Sync Wait"
                with m.Elif(timer == 0):
                    m.d.sync += measure.eq(0)
                    m.next = "Sync Result Low"

            with m.State("Sync Wait"):
                m.d.sync += timer.eq(timer - 1)
                with m.If(~bkgd_i):
                    m.next = "Sync Measure"
                with m.Elif(timer == 0):
                    # No response: report 0, which the host reads as "no target".
                    m.d.sync += measure.eq(0)
                    m.next = "Sync Result Low"

            with m.State("Sync Measure"):
                with m.If(bkgd_i):
                    m.next = "Sync Result Low"
                with m.Elif(measure.all()):
                    # Saturate rather than wrap, so an absurdly slow target reads as an error.
                    m.next = "Sync Result Low"
                with m.Else():
                    m.d.sync += measure.eq(measure + 1)

            with m.State("Sync Result Low"):
                m.d.comb += self.o_stream.payload.eq(measure[0:8])
                m.d.comb += self.o_stream.valid.eq(1)
                with m.If(self.o_stream.ready):
                    m.next = "Sync Result High"

            with m.State("Sync Result High"):
                m.d.comb += self.o_stream.payload.eq(measure[8:16])
                m.d.comb += self.o_stream.valid.eq(1)
                with m.If(self.o_stream.ready):
                    m.next = "Read Command"

        return m


class HCS08Error(GlasgowAppletError):
    pass


class BDCStatus(int):
    """Contents of the BDC status and control register (BDCSCR)."""

    ENBDM  = 0x80
    BDMACT = 0x40
    BKPTEN = 0x20
    FTS    = 0x10
    CLKSW  = 0x08
    WS     = 0x04
    WSF    = 0x02
    DVF    = 0x01

    @property
    def enbdm(self) -> bool:
        """BDM is enabled, so active background mode commands are permitted."""
        return bool(self & self.ENBDM)

    @property
    def bdmact(self) -> bool:
        """The target is in active background mode and waiting for serial commands."""
        return bool(self & self.BDMACT)

    @property
    def clksw(self) -> bool:
        """The BDC is clocked from the CPU bus clock rather than the alternate source."""
        return bool(self & self.CLKSW)

    @property
    def ws(self) -> bool:
        """The target CPU is in wait or stop mode."""
        return bool(self & self.WS)

    @property
    def wsf(self) -> bool:
        """The last memory access failed because the CPU entered wait or stop mode."""
        return bool(self & self.WSF)

    @property
    def dvf(self) -> bool:
        """The last memory access failed because a slow memory access was in progress."""
        return bool(self & self.DVF)

    def __str__(self):
        flags = [name for name, bit in (
            ("ENBDM", self.ENBDM), ("BDMACT", self.BDMACT), ("BKPTEN", self.BKPTEN),
            ("FTS", self.FTS), ("CLKSW", self.CLKSW), ("WS", self.WS),
            ("WSF", self.WSF), ("DVF", self.DVF),
        ) if self & bit]
        return f"{self:#04x}<{'|'.join(flags) if flags else '-'}>"


class _BDC(py_enum.IntEnum):
    """BDC command opcodes; see HCS08RMV1 Table 7-1."""

    ACK_ENABLE    = 0xD5
    ACK_DISABLE   = 0xD6
    BACKGROUND    = 0x90
    READ_STATUS   = 0xE4
    WRITE_CONTROL = 0xC4
    READ_BYTE     = 0xE0
    READ_BYTE_WS  = 0xE1
    READ_LAST     = 0xE8
    WRITE_BYTE    = 0xC0
    WRITE_BYTE_WS = 0xC1
    READ_BKPT     = 0xE2
    WRITE_BKPT    = 0xC2
    GO            = 0x08
    TRACE1        = 0x10
    TAGGO         = 0x18
    READ_A        = 0x68
    READ_CCR      = 0x69
    READ_PC       = 0x6B
    READ_HX       = 0x6C
    READ_SP       = 0x6F
    READ_NEXT     = 0x70
    READ_NEXT_WS  = 0x71
    WRITE_A       = 0x48
    WRITE_CCR     = 0x49
    WRITE_PC      = 0x4B
    WRITE_HX      = 0x4C
    WRITE_SP      = 0x4F
    WRITE_NEXT    = 0x50
    WRITE_NEXT_WS = 0x51


# FLASH module registers, common to the HCS08 family; see MC9S08AW60 §4.6.
FCDIV_addr = 0x1820
FOPT_addr  = 0x1821
FCNFG_addr = 0x1823
FPROT_addr = 0x1824
FSTAT_addr = 0x1825
FCMD_addr  = 0x1826

FSTAT_FCBEF   = 0x80
FSTAT_FCCF    = 0x40
FSTAT_FPVIOL  = 0x20
FSTAT_FACCERR = 0x10
FSTAT_FBLANK  = 0x04

FCMD_BLANK_CHECK   = 0x05
FCMD_BYTE_PROGRAM  = 0x20
FCMD_BURST_PROGRAM = 0x25
FCMD_PAGE_ERASE    = 0x40
FCMD_MASS_ERASE    = 0x41

FLASH_PAGE_SIZE = 512

# BDC commands pipelined into one USB transfer by `_burst`. Each memory access returns 2 bytes,
# so this keeps at most 256 bytes of response in flight, comfortably inside one 512-byte FX2 IN
# endpoint buffer. Larger batches buy nothing: past a few dozen commands the USB round trip is
# already amortised to insignificance against the BDC bit time.
_BURST_COMMANDS = 128


class DebugHCS08Interface:
    def __init__(self, logger: logging.Logger, assembly: AbstractAssembly, *,
                 bkgd: GlasgowPin, reset: GlasgowPin | None = None):
        self._logger = logger
        self._level  = logging.DEBUG if self._logger.name == __name__ else logging.TRACE

        # BKGD has an on-chip pullup on the target, but holding it high here too keeps the line
        # defined when no target is attached.
        assembly.use_pulls({bkgd: "high"})
        if reset is not None:
            assembly.use_pulls({reset: "high"})
        ports = assembly.add_port_group(bkgd=bkgd, reset=reset)
        component = assembly.add_submodule(
            DebugHCS08Component(ports, sys_clk_period=assembly.sys_clk_period))
        self._pipe = assembly.add_inout_pipe(component.o_stream, component.i_stream)
        self._sys_clk_period = assembly.sys_clk_period
        self._has_reset = reset is not None
        self._divisor = None

    def _log(self, message: str, *args):
        self._logger.log(self._level, "HCS08: " + message, *args)

    @property
    def bdc_clock_period(self) -> float | None:
        """Period of the target BDC clock, in seconds, or :py:`None` if :meth:`sync` has not run."""
        if self._divisor is None:
            return None
        return self._divisor * self._sys_clk_period

    # === Link layer ===

    async def sync(self) -> float:
        """Determine the target BDC communication speed.

        Drives the SYNC request and measures the target's 128-cycle response pulse, then configures
        the gateware bit timing to match. Returns the recovered BDC clock frequency, in Hz.
        """
        await self._pipe.send([_Command.Sync.value])
        await self._pipe.flush()
        count, = struct.unpack("<H", await self._pipe.recv(2))
        if count == 0:
            raise HCS08Error("no SYNC response from target; check BKGD wiring and target power")
        if count == 0xFFFF:
            raise HCS08Error("SYNC response too long to measure; target clock is implausibly slow")
        # The response is 128 BDC cycles, so one BDC cycle is count/128 sys clock cycles.
        divisor = max(1, round(count / _SYNC_RESPONSE))
        self._divisor = divisor
        await self._pipe.send(struct.pack("<BH", _Command.SetDivisor.value, divisor))
        await self._pipe.flush()
        frequency = 1.0 / (divisor * self._sys_clk_period)
        self._log(f"sync count={count} divisor={divisor} freq={frequency/1e6:.3f}MHz")
        return frequency

    def _encode(self, opcode: int, *,
                address: int | None = None, write: bytes = b"",
                delay: bool = False, status: bool = False, read: int = 0) -> tuple[bytes, int]:
        """Encode one BDC command per the coding structure of HCS08RMV1 Table 7-1.

        Returns the gateware opcodes implementing it and the number of bytes it will return.
        """
        if self._divisor is None:
            raise HCS08Error("BDC bit timing is not configured; call sync() first")
        seq = bytearray([_Command.Transmit.value, opcode])
        if address is not None:
            seq += bytes([_Command.Transmit.value, (address >> 8) & 0xFF,
                          _Command.Transmit.value, address & 0xFF])
        for byte in write:
            seq += bytes([_Command.Transmit.value, byte])
        if delay:
            seq += bytes([_Command.Delay.value])
        count = read + (1 if status else 0)
        seq += bytes([_Command.Receive.value]) * count
        return bytes(seq), count

    async def _command(self, opcode: int, **kwargs) -> bytes:
        """Issue one BDC command and return its response."""
        seq, count = self._encode(opcode, **kwargs)
        await self._pipe.send(seq)
        await self._pipe.flush()
        if count == 0:
            return b""
        data = bytes(await self._pipe.recv(count))
        if kwargs.get("status"):
            self._check_status(BDCStatus(data[0]), opcode)
            return data[1:]
        return data

    async def _burst(self, opcode: int, params: list[dict], response_len: int) -> list[bytes]:
        """Issue many identical-shaped BDC commands, sharing one USB round trip per batch.

        Only the ``_WS`` command variants may be used: the status byte each one reports is how
        a failure part-way through a batch is detected, since nothing else inspects the target
        between commands.

        A single BDC command costs well under a millisecond of bit time but a full USB round trip
        to set up, so issuing them one at a time is dominated by round trips rather than by the
        target. Pipelining a batch into one transfer removes that. Batches are sized so the
        responses in flight stay far below the 512-byte FX2 IN endpoint: the component cannot
        stall for want of somewhere to put them, which would otherwise deadlock against `send`.
        """
        results = []
        for start in range(0, len(params), _BURST_COMMANDS):
            batch = params[start:start + _BURST_COMMANDS]
            seq = bytearray()
            for kwargs in batch:
                ops, count = self._encode(opcode, **kwargs)
                assert count == response_len
                seq += ops
            await self._pipe.send(seq)
            await self._pipe.flush()
            data = bytes(await self._pipe.recv(len(batch) * response_len))
            for index in range(len(batch)):
                response = data[index * response_len:(index + 1) * response_len]
                self._check_status(BDCStatus(response[0]), opcode)
                results.append(response[1:])
        return results

    def _check_status(self, status: BDCStatus, opcode: int):
        if status.wsf:
            raise HCS08Error(f"BDC command {opcode:#04x} failed: target entered wait or stop mode "
                             f"(status {status})")
        if status.dvf:
            raise HCS08Error(f"BDC command {opcode:#04x} failed: slow memory access in progress "
                             f"(status {status})")

    # === Non-intrusive commands ===

    async def read_status(self) -> BDCStatus:
        """Read the BDC status and control register (BDCSCR)."""
        data = await self._command(_BDC.READ_STATUS, read=1)
        status = BDCStatus(data[0])
        self._log(f"read status={status}")
        return status

    async def write_control(self, value: int):
        """Write the control bits of the BDC status and control register (BDCSCR)."""
        self._log(f"write control={BDCStatus(value)}")
        await self._command(_BDC.WRITE_CONTROL, write=bytes([value]))

    async def ack_disable(self):
        """Disable the hardware handshake protocol.

        This applet uses fixed timing rather than ACK pulses, so the handshake is disabled at
        startup in case a previous session enabled it.
        """
        await self._command(_BDC.ACK_DISABLE, delay=True)

    async def background(self):
        """Request entry into active background mode.

        Ignored by the target unless ``ENBDM`` has been set with :meth:`write_control`.
        """
        self._log("background")
        await self._command(_BDC.BACKGROUND, delay=True)

    async def read_byte(self, address: int) -> int:
        """Read one byte of target memory, checking the resulting status."""
        assert address in range(0x10000)
        data = await self._command(_BDC.READ_BYTE_WS, address=address,
                                   delay=True, status=True, read=1)
        self._log(f"read {address:#06x}={data[0]:#04x}")
        return data[0]

    async def write_byte(self, address: int, value: int):
        """Write one byte of target memory, checking the resulting status."""
        assert address in range(0x10000) and value in range(0x100)
        self._log(f"write {address:#06x}={value:#04x}")
        await self._command(_BDC.WRITE_BYTE_WS, address=address, write=bytes([value]),
                            delay=True, status=True)

    async def read(self, address: int, length: int) -> bytes:
        """Read ``length`` bytes of target memory starting at ``address``."""
        assert address in range(0x10000) and address + length <= 0x10000
        # READ_NEXT_WS would shave two byte times per access, but it requires active background
        # mode and clobbers H:X, so plain READ_BYTE_WS is used to keep reads usable while
        # the target is running.
        self._log(f"read {address:#06x} length={length}")
        responses = await self._burst(_BDC.READ_BYTE_WS, [
            dict(address=address + offset, delay=True, status=True, read=1)
            for offset in range(length)
        ], response_len=2)
        return b"".join(responses)

    async def write(self, address: int, data: bytes):
        """Write ``data`` to target memory starting at ``address``."""
        assert address in range(0x10000) and address + len(data) <= 0x10000
        self._log(f"write {address:#06x} length={len(data)}")
        await self._burst(_BDC.WRITE_BYTE_WS, [
            dict(address=address + offset, write=bytes([byte]), delay=True, status=True)
            for offset, byte in enumerate(data)
        ], response_len=1)

    async def read_bkpt(self) -> int:
        """Read the BDC breakpoint match register (BDCBKPT)."""
        data = await self._command(_BDC.READ_BKPT, read=2)
        return int.from_bytes(data, "big")

    async def write_bkpt(self, address: int):
        """Write the BDC breakpoint match register (BDCBKPT)."""
        assert address in range(0x10000)
        await self._command(_BDC.WRITE_BKPT, write=address.to_bytes(2, "big"))

    # === Active background mode commands ===

    async def _require_active(self):
        status = await self.read_status()
        if not status.bdmact:
            raise HCS08Error(f"target is not in active background mode (status {status})")

    async def go(self):
        """Resume execution of the user program at the current PC."""
        self._log("go")
        await self._command(_BDC.GO, delay=True)

    async def trace1(self):
        """Execute one instruction, then return to active background mode."""
        self._log("trace1")
        await self._command(_BDC.TRACE1, delay=True)

    async def read_a(self) -> int:
        """Read the accumulator (A)."""
        return (await self._command(_BDC.READ_A, delay=True, read=1))[0]

    async def write_a(self, value: int):
        """Write the accumulator (A)."""
        assert value in range(0x100)
        await self._command(_BDC.WRITE_A, write=bytes([value]), delay=True)

    async def read_ccr(self) -> int:
        """Read the condition code register (CCR)."""
        return (await self._command(_BDC.READ_CCR, delay=True, read=1))[0]

    async def write_ccr(self, value: int):
        """Write the condition code register (CCR)."""
        assert value in range(0x100)
        await self._command(_BDC.WRITE_CCR, write=bytes([value]), delay=True)

    async def read_pc(self) -> int:
        """Read the program counter (PC)."""
        return int.from_bytes(await self._command(_BDC.READ_PC, delay=True, read=2), "big")

    async def write_pc(self, value: int):
        """Write the program counter (PC)."""
        assert value in range(0x10000)
        await self._command(_BDC.WRITE_PC, write=value.to_bytes(2, "big"), delay=True)

    async def read_hx(self) -> int:
        """Read the H:X index register pair."""
        return int.from_bytes(await self._command(_BDC.READ_HX, delay=True, read=2), "big")

    async def write_hx(self, value: int):
        """Write the H:X index register pair."""
        assert value in range(0x10000)
        await self._command(_BDC.WRITE_HX, write=value.to_bytes(2, "big"), delay=True)

    async def read_sp(self) -> int:
        """Read the stack pointer (SP)."""
        return int.from_bytes(await self._command(_BDC.READ_SP, delay=True, read=2), "big")

    async def write_sp(self, value: int):
        """Write the stack pointer (SP)."""
        assert value in range(0x10000)
        await self._command(_BDC.WRITE_SP, write=value.to_bytes(2, "big"), delay=True)

    async def read_registers(self) -> dict[str, int]:
        """Read all CPU registers. Requires active background mode."""
        await self._require_active()
        return {
            "PC":  await self.read_pc(),
            "SP":  await self.read_sp(),
            "H:X": await self.read_hx(),
            "A":   await self.read_a(),
            "CCR": await self.read_ccr(),
        }

    # === Target control ===

    async def set_reset(self, asserted: bool):
        """Drive the target RESET pin low, or release it."""
        if not self._has_reset:
            raise HCS08Error("no RESET pin was assigned; pass --reset to use target reset")
        await self._pipe.send([_Command.SetReset.value, 0 if asserted else 1])
        await self._pipe.flush()

    async def set_bkgd(self, level: int | None):
        """Hold BKGD at ``level``, or release it if ``level`` is :py:`None`.

        Only meaningful between transfers; any transfer overrides the held level for its duration
        and the held level is restored afterwards.
        """
        await self._pipe.send([_Command.SetBkgd.value, 2 if level is None else level])
        await self._pipe.flush()

    async def reset_into_bdm(self):
        """Reset the target into active background mode.

        Holds BKGD low across the rising edge of RESET, which selects active background mode
        instead of the user application program. This is the only way to gain control of a target
        whose FLASH is blank or whose program disables the BDC.
        """
        if not self._has_reset:
            raise HCS08Error("no RESET pin was assigned; pass --reset to use target reset")
        self._log("reset into BDM")
        # Each step is flushed separately so that the ordering (and thus the level of BKGD at
        # the rising edge of RESET) is determined by these awaits rather than by FIFO occupancy.
        await self.set_bkgd(0)
        await self.set_reset(True)
        await asyncio.sleep(0.01)
        await self.set_reset(False)
        await asyncio.sleep(0.01)
        await self.set_bkgd(None)
        await asyncio.sleep(0.001)
        await self.sync()

    async def reset_into_user(self):
        """Reset the target into the normal user program."""
        if not self._has_reset:
            raise HCS08Error("no RESET pin was assigned; pass --reset to use target reset")
        self._log("reset into user mode")
        await self.set_bkgd(None)
        await self.set_reset(True)
        await asyncio.sleep(0.01)
        await self.set_reset(False)
        await asyncio.sleep(0.01)

    async def enable_bdm(self):
        """Enable active background mode and halt the target."""
        status = await self.read_status()
        if not status.enbdm:
            await self.write_control(status | BDCStatus.ENBDM)
        if not (await self.read_status()).bdmact:
            await self.background()

    # === FLASH programming ===

    async def flash_init(self, bus_frequency: float):
        """Configure the FLASH clock divider for a given target bus frequency, in Hz.

        The FLASH command state machine requires a 150..200 kHz internal clock; see MC9S08AW60
        §4.6.1. ``FCDIV`` is write-once per reset, so this is a no-op if it is already loaded.
        """
        fcdiv = await self.read_byte(FCDIV_addr)
        if fcdiv & 0x80: # DIVLD
            self._log(f"flash divider already loaded FCDIV={fcdiv:#04x}")
            return
        # Clear any stale error flag; FCDIV cannot be written while FACCERR is set.
        await self._flash_clear_errors()
        prdiv8, div = self._flash_divider(bus_frequency)
        value = (0x40 if prdiv8 else 0x00) | div
        self._log(f"flash divider bus={bus_frequency/1e6:.3f}MHz FCDIV={value:#04x}")
        await self.write_byte(FCDIV_addr, value)
        if not (await self.read_byte(FCDIV_addr)) & 0x80:
            raise HCS08Error("failed to load FCDIV; target bus clock may be stopped")

    @staticmethod
    def _flash_divider(bus_frequency: float) -> tuple[bool, int]:
        """Pick ``PRDIV8`` and ``DIV[5:0]`` yielding an FCLK within 150..200 kHz."""
        for prdiv8 in (False, True):
            prescale = 8 if prdiv8 else 1
            for div in range(64):
                fclk = bus_frequency / (prescale * (div + 1))
                if 150e3 <= fclk <= 200e3:
                    return prdiv8, div
        raise HCS08Error(f"cannot derive a 150..200 kHz FLASH clock from a "
                         f"{bus_frequency/1e6:.3f} MHz bus clock")

    async def _flash_clear_errors(self):
        fstat = await self.read_byte(FSTAT_addr)
        if fstat & (FSTAT_FACCERR | FSTAT_FPVIOL):
            await self.write_byte(FSTAT_addr, fstat & (FSTAT_FACCERR | FSTAT_FPVIOL))

    async def flash_unprotect(self):
        """Disable FLASH block protection for the current session."""
        self._log("flash unprotect")
        await self.write_byte(FPROT_addr, 0xFF)

    async def _flash_command(self, address: int, value: int, command: int, *, timeout: int = 1000):
        """Run the three-step FLASH command sequence of MC9S08AW60 §4.4.3."""
        await self._flash_clear_errors()
        fstat = await self.read_byte(FSTAT_addr)
        if not fstat & FSTAT_FCBEF:
            raise HCS08Error(f"FLASH command buffer not empty (FSTAT {fstat:#04x})")
        # 1. Latch address and data by writing to the FLASH array.
        await self.write_byte(address, value)
        # 2. Latch the command code.
        await self.write_byte(FCMD_addr, command)
        # 3. Clear FCBEF to launch the command.
        await self.write_byte(FSTAT_addr, FSTAT_FCBEF)
        for _ in range(timeout):
            fstat = await self.read_byte(FSTAT_addr)
            if fstat & FSTAT_FACCERR:
                raise HCS08Error(f"FLASH access error at {address:#06x} "
                                 f"(command {command:#04x})")
            if fstat & FSTAT_FPVIOL:
                raise HCS08Error(f"FLASH protection violation at {address:#06x} "
                                 f"(command {command:#04x}); try flash_unprotect()")
            if fstat & FSTAT_FCCF:
                return
        raise HCS08Error(f"FLASH command {command:#04x} at {address:#06x} did not complete")

    async def flash_mass_erase(self):
        """Erase the entire FLASH array."""
        self._log("flash mass erase")
        await self.flash_unprotect()
        await self._flash_command(0xFFFE, 0xFF, FCMD_MASS_ERASE, timeout=10000)

    async def flash_page_erase(self, address: int):
        """Erase the 512-byte FLASH page containing ``address``."""
        self._log(f"flash page erase {address:#06x}")
        await self._flash_command(address, 0xFF, FCMD_PAGE_ERASE, timeout=10000)

    async def flash_program(self, address: int, data: bytes):
        """Program ``data`` into FLASH starting at ``address``.

        The page containing each address must have been erased first; the HCS08 FLASH does not
        permit programming a byte twice without an intervening erase.
        """
        self._log(f"flash program {address:#06x} length={len(data)}")
        for offset, byte in enumerate(data):
            if byte != 0xFF: # an erased byte is already 0xFF, so skip the command entirely
                await self._flash_command(address + offset, byte, FCMD_BYTE_PROGRAM)

    async def flash_verify(self, address: int, data: bytes):
        """Read back and compare ``data`` against FLASH starting at ``address``."""
        actual = await self.read(address, len(data))
        for offset, (want, got) in enumerate(zip(data, actual)):
            if want != got:
                raise HCS08Error(f"verification failed at {address + offset:#06x}: "
                                 f"expected {want:#04x}, read {got:#04x}")


def parse_srecord(text: str) -> dict[int, int]:
    """Parse Motorola S-record text into a mapping of address to byte value."""
    memory = {}
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        if not line.startswith("S"):
            raise HCS08Error(f"line {lineno}: not an S-record")
        try:
            record = bytes.fromhex(line[2:])
        except ValueError:
            raise HCS08Error(f"line {lineno}: malformed hexadecimal data") from None
        kind = line[1]
        if kind in "789":  # start address records carry no data
            continue
        if kind not in "123":
            continue
        if len(record) < 1 or record[0] != len(record) - 1:
            raise HCS08Error(f"line {lineno}: incorrect S-record length field")
        if (sum(record[:-1]) + record[-1]) & 0xFF != 0xFF:
            raise HCS08Error(f"line {lineno}: S-record checksum mismatch")
        addr_len = {"1": 2, "2": 3, "3": 4}[kind]
        address = int.from_bytes(record[1:1 + addr_len], "big")
        for offset, byte in enumerate(record[1 + addr_len:-1]):
            memory[address + offset] = byte
    if not memory:
        raise HCS08Error("S-record file contains no data records")
    return memory


def coalesce(memory: dict[int, int]) -> list[tuple[int, bytes]]:
    """Group a sparse address-to-byte mapping into contiguous ``(address, data)`` chunks."""
    chunks = []
    for address in sorted(memory):
        if chunks and address == chunks[-1][0] + len(chunks[-1][1]):
            chunks[-1][1].append(memory[address])
        else:
            chunks.append((address, bytearray([memory[address]])))
    return [(address, bytes(data)) for address, data in chunks]


class DebugHCS08Applet(GlasgowAppletV2):
    preview = True
    logger = logging.getLogger(__name__)
    help = "program and debug Freescale/NXP HCS08 MCUs via BDM"
    description = """
    Program and debug HCS08 (and compatible S08) microcontrollers through the single-wire
    background debug mode (BDM) interface on the BKGD pin.

    The BDC has no clock line; bit timing is referenced to the target's own BDC clock. The applet
    recovers that clock automatically with a SYNC request at startup, so no frequency needs to be
    specified for debugging.

    Connect BKGD to pin 1 of the standard 6-pin BDM header and ground to pin 2. Connecting RESET
    (pin 4) is optional but required to halt a target whose FLASH is blank or whose program
    disables the BDC, since that needs BKGD held low across the rising edge of RESET.

    FLASH programming additionally needs the target bus frequency (--bus-frequency), because the
    FLASH command state machine must be clocked between 150 and 200 kHz and the divider that
    achieves this cannot be derived from the BDC clock.
    """
    # BKGD is a pseudo-open-drain signal; the revA/B level shifters interfere with it.
    required_revision = "C0"
    hcs08_iface: DebugHCS08Interface

    @classmethod
    def add_build_arguments(cls, parser, access):
        access.add_voltage_argument(parser)
        access.add_pins_argument(parser, "bkgd", required=True, default=True)
        access.add_pins_argument(parser, "reset")

    def build(self, args):
        with self.assembly.add_applet(self):
            self.assembly.use_voltage(args.voltage)
            self.hcs08_iface = DebugHCS08Interface(self.logger, self.assembly,
                bkgd=args.bkgd, reset=args.reset)

    @classmethod
    def add_setup_arguments(cls, parser):
        parser.add_argument(
            "--reset-into-bdm", default=False, action="store_true",
            help="reset the target into active background mode on startup (requires --reset)")

    async def setup(self, args):
        if args.reset_into_bdm:
            await self.hcs08_iface.reset_into_bdm()
        else:
            await self.hcs08_iface.sync()
        await self.hcs08_iface.ack_disable()

    @classmethod
    def add_run_arguments(cls, parser):
        def address(value):
            return int(value, 0)
        def length(value):
            return int(value, 0)

        p_operation = parser.add_subparsers(dest="operation", metavar="OPERATION", required=True)

        p_operation.add_parser(
            "status", help="report BDC status and, if halted, CPU registers")

        p_operation.add_parser(
            "halt", help="enable BDM and halt the target")

        p_operation.add_parser(
            "run", help="resume execution of the user program")

        p_read = p_operation.add_parser(
            "read", help="read target memory")
        p_read.add_argument(
            "address", metavar="ADDRESS", type=address,
            help="starting address")
        p_read.add_argument(
            "length", metavar="LENGTH", type=length,
            help="number of bytes to read")
        p_read.add_argument(
            "-f", "--file", metavar="FILE", type=argparse.FileType("wb"),
            help="write data to FILE instead of displaying it")

        p_write = p_operation.add_parser(
            "write", help="write target memory (RAM and registers only)")
        p_write.add_argument(
            "address", metavar="ADDRESS", type=address,
            help="starting address")
        p_write.add_argument(
            "data", metavar="DATA", type=lambda value: bytes.fromhex(value),
            help="hexadecimal data to write")

        p_erase = p_operation.add_parser(
            "erase", help="mass erase the FLASH array")

        p_program = p_operation.add_parser(
            "program", help="erase and program FLASH from an S-record file")
        p_program.add_argument(
            "file", metavar="FILE", type=argparse.FileType("r"),
            help="S-record (S19) file to program")
        p_program.add_argument(
            "--bus-frequency", metavar="FREQ", type=float, required=True,
            help="target bus clock frequency, in MHz")
        p_program.add_argument(
            "--no-verify", default=False, action="store_true",
            help="skip read-back verification")

    async def run(self, args):
        iface = self.hcs08_iface

        match args.operation:
            case "status":
                status = await iface.read_status()
                self.logger.info("BDC status: %s", status)
                self.logger.info("BDC clock: %.3f MHz",
                                 1e-6 / iface.bdc_clock_period)
                if status.bdmact:
                    registers = await iface.read_registers()
                    self.logger.info("PC=%04X SP=%04X H:X=%04X A=%02X CCR=%02X",
                                     registers["PC"], registers["SP"], registers["H:X"],
                                     registers["A"], registers["CCR"])
                else:
                    self.logger.info("target is running; use `halt` to stop it")

            case "halt":
                await iface.enable_bdm()
                registers = await iface.read_registers()
                self.logger.info("halted at PC=%04X SP=%04X H:X=%04X A=%02X CCR=%02X",
                                 registers["PC"], registers["SP"], registers["H:X"],
                                 registers["A"], registers["CCR"])

            case "run":
                await iface.go()
                self.logger.info("target resumed")

            case "read":
                data = await iface.read(args.address, args.length)
                if args.file:
                    args.file.write(data)
                else:
                    for offset in range(0, len(data), 16):
                        row = data[offset:offset + 16]
                        hexed = " ".join(f"{byte:02x}" for byte in row)
                        ascii_ = "".join(chr(b) if 0x20 <= b < 0x7F else "." for b in row)
                        print(f"{args.address + offset:04x}  {hexed:<47}  {ascii_}")

            case "write":
                await iface.write(args.address, args.data)
                self.logger.info("wrote %d byte(s) at %#06x", len(args.data), args.address)

            case "erase":
                await iface.enable_bdm()
                await iface.flash_mass_erase()
                self.logger.info("FLASH mass erased")

            case "program":
                chunks = coalesce(parse_srecord(args.file.read()))
                total = sum(len(data) for _, data in chunks)
                await iface.enable_bdm()
                await iface.flash_init(args.bus_frequency * 1e6)
                await iface.flash_unprotect()
                await iface.flash_mass_erase()
                for address, data in chunks:
                    self.logger.info("programming %#06x..%#06x",
                                     address, address + len(data) - 1)
                    await iface.flash_program(address, data)
                if not args.no_verify:
                    for address, data in chunks:
                        await iface.flash_verify(address, data)
                    self.logger.info("verified %d byte(s)", total)
                self.logger.info("programmed %d byte(s)", total)

    @classmethod
    def tests(cls):
        from . import test
        return test.DebugHCS08AppletTestCase
