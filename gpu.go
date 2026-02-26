// gpu.go — Go-wrapper voor de CUDA simulatie-DLL
//
// Laadt simulate.dll dynamisch bij opstart (syscall.LoadDLL).
// Als de DLL niet beschikbaar is, valt de engine terug op CPU-simulatie.
//
// SimState moet qua geheugenopbouw exact overeenkomen met de C-struct in simulate.cu.
// Alle velden zijn 1 byte, dus geen alignment-padding.

package main

import (
	"fmt"
	"math/rand"
	"sync"
	"syscall"
	"unsafe"
)

// ---------------------------------------------------------------------------
// SimState — 76 bytes, exact gelijk aan de C-struct in simulate.cu
// ---------------------------------------------------------------------------

// Rang-indices: 0=3, 1=4, ..., 10=K, 11=A(1), 12=2(wild), 13=Joker(reset)
// Formule: gpuRankIdx = goRank - 3  (werkt voor 3→14, 15→12, 16→13)

const (
	gpuWildIdx  = 12
	gpuJokerIdx = 13
)

type SimState struct {
	HandCounts   [4][14]uint8 // [speler][rang] aantal kaarten
	TableRankIdx int8         // -1+tableCount==0→open, -1+tableCount>0→wild-gesloten, >=0→gesloten
	TableCount   uint8        // benodigde kaartenaantal
	ConsecPasses uint8        // opeenvolgende passen
	CurrentTurn  uint8        // huidige speler
	NumPlayers   uint8        // aantal spelers
	LastPlayerID int8         // wie laast speelde (-1 = niemand)
	NumFinished  uint8        // afgeronde spelers
	FinishRank   [4]uint8     // eindpositie per speler (255 = nog niet klaar)
	Finished     [4]uint8     // 1 = klaar
	GameOver     uint8        // 1 = spel voorbij
	MyID         uint8        // speler voor wie gescoord wordt
	Pad          [3]uint8     // padding → totaal 76 bytes
}

func init() {
	// Controleer of de Go-struct exact 76 bytes is (zelfde als C)
	if sz := int(unsafe.Sizeof(SimState{})); sz != 76 {
		panic(fmt.Sprintf("SimState grootte mismatch: Go=%d, verwacht=76", sz))
	}
}

// ---------------------------------------------------------------------------
// DLL laden
// ---------------------------------------------------------------------------

var (
	gpuDLL         *syscall.DLL
	procDevCount   *syscall.Proc
	procStateSize  *syscall.Proc
	procBatchSim   *syscall.Proc
	gpuEnabled     bool
	gpuMu          sync.Mutex
	gpuSeedCounter uint64
)

func gpuInit() {
	dll, err := syscall.LoadDLL("simulate.dll")
	if err != nil {
		// DLL niet gevonden — gebruik CPU
		return
	}

	procDevCount, err = dll.FindProc("gpu_device_count")
	if err != nil {
		dll.Release()
		return
	}
	procStateSize, err = dll.FindProc("gpu_sim_state_size")
	if err != nil {
		dll.Release()
		return
	}
	procBatchSim, err = dll.FindProc("gpu_batch_simulate")
	if err != nil {
		dll.Release()
		return
	}

	// Controleer of er een CUDA-apparaat beschikbaar is
	rDevCount, _, _ := procDevCount.Call()
	devCount := int32(rDevCount)
	if devCount <= 0 {
		dll.Release()
		return
	}

	// Controleer of de DLL-structuur overeenkomt
	rSize, _, _ := procStateSize.Call()
	dllSize := int(int32(rSize))
	goSize := int(unsafe.Sizeof(SimState{}))
	if dllSize != goSize {
		fmt.Printf("[GPU] Waarschuwing: SimState mismatch (DLL=%d bytes, Go=%d bytes) — GPU uitgeschakeld\n",
			dllSize, goSize)
		dll.Release()
		return
	}

	gpuDLL = dll
	gpuEnabled = true
	gpuSeedCounter = uint64(rand.Int63())
	fmt.Printf("[GPU] RTX 4080 GPU gedetecteerd (%d apparaat) — GPU-simulatie actief\n", devCount)
}

// ---------------------------------------------------------------------------
// GameState → SimState converteren
// ---------------------------------------------------------------------------

func rankToGPUIdx(r Rank) int {
	return int(r) - 3 // 3→0, 4→1, ..., 14→11, 15→12, 16→13
}

func gameStateToSimState(gs *GameState, myID int) SimState {
	var s SimState
	s.NumPlayers  = uint8(gs.NumPlayers)
	s.CurrentTurn = uint8(gs.CurrentTurn)
	s.MyID        = uint8(myID)
	s.LastPlayerID = -1

	// Handen en eindposities
	for p := 0; p < gs.NumPlayers; p++ {
		s.FinishRank[p] = 255 // nog niet klaar
		for _, c := range gs.Hands[p].Cards {
			idx := rankToGPUIdx(c.Rank)
			if idx >= 0 && idx < 14 {
				s.HandCounts[p][idx]++
			}
		}
		if gs.Finished[p] {
			s.Finished[p] = 1
		}
	}

	// Eindposities uit de Ranking-slice
	for i, pid := range gs.Ranking {
		if pid >= 0 && pid < gs.NumPlayers {
			s.FinishRank[pid] = uint8(i)
		}
	}
	s.NumFinished = uint8(len(gs.Ranking))

	if gs.GameOver {
		s.GameOver = 1
	}

	// Tafeltoestand
	if gs.Round.IsOpen {
		s.TableRankIdx = -1
		s.TableCount   = 0
	} else {
		idx := rankToGPUIdx(gs.Round.TableRank)
		if idx < 0 {
			idx = -1 // onverwacht, behandel als wild-gesloten
		}
		s.TableRankIdx = int8(idx)
		s.TableCount   = uint8(gs.Round.Count)
	}
	s.ConsecPasses = uint8(gs.Round.ConsecPasses)
	if gs.Round.LastPlayerID >= 0 {
		s.LastPlayerID = int8(gs.Round.LastPlayerID)
	}

	return s
}

// ---------------------------------------------------------------------------
// GPU batch-simulatie aanroepen
// ---------------------------------------------------------------------------

const gpuBatchSize = 256

func gpuBatchSimulate(states []SimState, seed uint64) []float64 {
	n := len(states)
	if n == 0 {
		return nil
	}

	results := make([]float32, n)

	gpuMu.Lock()
	procBatchSim.Call(
		uintptr(unsafe.Pointer(&states[0])),
		uintptr(unsafe.Pointer(&results[0])),
		uintptr(n),
		uintptr(seed),
	)
	gpuMu.Unlock()

	out := make([]float64, n)
	for i, r := range results {
		out[i] = float64(r)
	}
	return out
}

// gpuNextSeed geeft een unieke seed terug voor elke GPU-aanroep.
func gpuNextSeed() uint64 {
	gpuMu.Lock()
	gpuSeedCounter += 1337
	s := gpuSeedCounter
	gpuMu.Unlock()
	return s
}
