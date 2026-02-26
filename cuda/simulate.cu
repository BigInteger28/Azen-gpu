// simulate.cu — CUDA kernel voor batch IS-MCTS rollouts (AZEN kaartspel)
// Compileer: zie build_gpu.bat
//
// Rangindices: 0=3, 1=4, 2=5, 3=6, 4=7, 5=8, 6=9, 7=10(X), 8=J, 9=Q, 10=K, 11=A(1)
//              12=2(wildcard), 13=Joker(reset)

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdint.h>

#define MAX_PLAYERS 4
#define NUM_RANKS   14
#define WILD_IDX    12   // 2 = wildcard
#define JOKER_IDX   13   // 0 = reset-kaart
#define MAX_COMBO   6    // max kaarten per zet

// ---------------------------------------------------------------------------
// SimState — gedeelde struct tussen Go en CUDA (exact 76 bytes, geen padding)
// Go-equivalent staat in gpu.go
// ---------------------------------------------------------------------------
#pragma pack(push, 1)
typedef struct {
    uint8_t handCounts[MAX_PLAYERS][NUM_RANKS]; // [speler][rang] = aantal, offset 0, 56 bytes
    int8_t  tableRankIdx;   // -1 + tableCount==0 → open ronde; -1 + tableCount>0 → wild-gesloten
                            // >=0 + tableCount>0 → normaal gesloten
    uint8_t tableCount;     // aantal kaarten nodig om te antwoorden
    uint8_t consecPasses;   // opeenvolgende passen in huidige ronde
    uint8_t currentTurn;    // spelersindex die aan de beurt is
    uint8_t numPlayers;     // aantal spelers (2–4)
    int8_t  lastPlayerID;   // wie laast speelde (-1 = niemand)
    uint8_t numFinished;    // aantal spelers dat klaar is
    uint8_t finishRank[MAX_PLAYERS]; // eindpositie per speler (255 = nog niet klaar)
    uint8_t finished[MAX_PLAYERS];   // 1 = klaar
    uint8_t gameOver;       // 1 = spel afgelopen
    uint8_t myID;           // speler voor wie we scoren
    uint8_t _pad[3];        // padding → totaal 76 bytes
} SimState;
#pragma pack(pop)

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__
int gpu_totalCards(const SimState* s, int pid) {
    int total = 0;
    for (int r = 0; r < NUM_RANKS; r++) total += s->handCounts[pid][r];
    return total;
}

__device__ __forceinline__
int gpu_activePlayers(const SimState* s) {
    int count = 0;
    for (int i = 0; i < (int)s->numPlayers; i++)
        if (!s->finished[i]) count++;
    return count;
}

__device__ __forceinline__
int gpu_nextActive(const SimState* s, int pid) {
    for (int i = 1; i <= (int)s->numPlayers; i++) {
        int next = (pid + i) % (int)s->numPlayers;
        if (!s->finished[next]) return next;
    }
    return pid;
}

__device__
void gpu_finishPlayer(SimState* s, int pid) {
    if (s->finished[pid]) return;
    s->finished[pid] = 1;
    s->finishRank[pid] = s->numFinished++;

    // Controleer of nog ≤1 speler over is → spel afgelopen
    int remaining = 0;
    for (int i = 0; i < (int)s->numPlayers; i++)
        if (!s->finished[i]) remaining++;

    if (remaining <= 1) {
        for (int i = 0; i < (int)s->numPlayers; i++) {
            if (!s->finished[i]) {
                s->finished[i] = 1;
                s->finishRank[i] = s->numFinished++;
            }
        }
        s->gameOver = 1;
    }
}

// ---------------------------------------------------------------------------
// Zet kiezen: geeft -1 terug voor pas, anders rankIdx*100 + numCards
// ---------------------------------------------------------------------------
__device__
int gpu_pickMove(SimState* s, curandState* rng) {
    int pid   = (int)s->currentTurn;
    int wilds  = (int)s->handCounts[pid][WILD_IDX];
    int jokers = (int)s->handCounts[pid][JOKER_IDX];

    int validRanks[NUM_RANKS];
    int validN = 0;

    int isOpen      = (s->tableRankIdx < 0 && s->tableCount == 0);
    int isWildClose = (s->tableRankIdx < 0 && s->tableCount > 0);
    int isClosed    = (s->tableRankIdx >= 0);

    if (isOpen) {
        // Vrije ronde: elke rang is geldig
        for (int r = 0; r < NUM_RANKS; r++)
            if (s->handCounts[pid][r] > 0)
                validRanks[validN++] = r;
    } else {
        // Gesloten ronde (normaal of wild-gesloten)
        int need    = (int)s->tableCount;
        int minRank = isClosed ? (int)s->tableRankIdx + 1 : 0;

        // Normale rangen die de tafel kunnen verslaan
        for (int r = minRank; r <= 11; r++) {
            int normals = (int)s->handCounts[pid][r];
            if (normals > 0 && normals + wilds >= need)
                validRanks[validN++] = r;
        }
        // Wild-only zet (verslaat niet, maar is geldig)
        if (wilds >= need)
            validRanks[validN++] = WILD_IDX;
        // Joker reset
        if (jokers > 0)
            validRanks[validN++] = JOKER_IDX;
    }

    // 30% kans op passen, of forceer als geen zet mogelijk
    if (validN == 0 || curand_uniform(rng) < 0.30f)
        return -1;

    // Kies willekeurige rang
    int choice  = (int)(curand_uniform(rng) * (float)validN);
    if (choice >= validN) choice = validN - 1;
    int rankIdx = validRanks[choice];

    // Bepaal aantal kaarten
    int numCards;
    if (rankIdx == JOKER_IDX) {
        numCards = 1;
    } else if (isOpen) {
        // Vrije ronde: 1 t/m min(beschikbaar, 6) kaarten
        int avail;
        if (rankIdx == WILD_IDX) {
            avail = wilds;
        } else {
            avail = (int)s->handCounts[pid][rankIdx] + wilds;
        }
        if (avail > MAX_COMBO) avail = MAX_COMBO;
        if (avail < 1) avail = 1;
        numCards = 1 + (int)(curand_uniform(rng) * (float)avail);
        if (numCards > avail) numCards = avail;
    } else {
        // Gesloten ronde: exact tableCount
        numCards = (int)s->tableCount;
    }

    return rankIdx * 100 + numCards;
}

// ---------------------------------------------------------------------------
// Zet toepassen
// ---------------------------------------------------------------------------
__device__
void gpu_applyMove(SimState* s, int move) {
    int pid = (int)s->currentTurn;

    if (move == -1) {
        // Pas
        s->consecPasses++;
        int active    = gpu_activePlayers(s);
        int threshold = active > 1 ? active - 1 : 1;

        if ((int)s->consecPasses >= threshold) {
            // Ronde resetten: de speler die laast speelde opent opnieuw
            int lastPID = (int)s->lastPlayerID;
            s->tableRankIdx = -1;
            s->tableCount   = 0;
            s->consecPasses = 0;
            if (lastPID >= 0 && !s->finished[lastPID]) {
                s->currentTurn = (uint8_t)lastPID;
            } else if (lastPID >= 0) {
                s->currentTurn = (uint8_t)gpu_nextActive(s, lastPID);
            } else {
                s->currentTurn = (uint8_t)gpu_nextActive(s, pid);
            }
        } else {
            s->currentTurn = (uint8_t)gpu_nextActive(s, pid);
        }
        return;
    }

    int rankIdx  = move / 100;
    int numCards = move % 100;

    if (rankIdx == JOKER_IDX) {
        // Joker reset
        s->handCounts[pid][JOKER_IDX]--;
        s->tableRankIdx = -1;
        s->tableCount   = 0;
        s->consecPasses = 0;
        s->lastPlayerID = (int8_t)pid;

        if (gpu_totalCards(s, pid) == 0) {
            gpu_finishPlayer(s, pid);
            if (s->gameOver) return;
            s->currentTurn = (uint8_t)gpu_nextActive(s, pid);
        }
        // anders: currentTurn blijft pid (joker-speler opent de volgende ronde)
        return;
    }

    // Normale of wild-only zet
    if (rankIdx == WILD_IDX) {
        // Wild-only: verwijder wilds, tableRankIdx ongewijzigd
        s->handCounts[pid][WILD_IDX] -= (uint8_t)numCards;
    } else {
        // Normaal (eventueel met wilds): stel tableRankIdx in op nieuwe rang
        int avail      = (int)s->handCounts[pid][rankIdx];
        int useNormals = (numCards < avail) ? numCards : avail;
        int useWilds   = numCards - useNormals;
        s->handCounts[pid][rankIdx]  -= (uint8_t)useNormals;
        s->handCounts[pid][WILD_IDX] -= (uint8_t)useWilds;
        s->tableRankIdx = (int8_t)rankIdx;
    }

    s->tableCount   = (uint8_t)numCards;
    s->consecPasses = 0;
    s->lastPlayerID = (int8_t)pid;

    if (gpu_totalCards(s, pid) == 0) {
        gpu_finishPlayer(s, pid);
        if (s->gameOver) return;
    }

    s->currentTurn = (uint8_t)gpu_nextActive(s, pid);
}

// ---------------------------------------------------------------------------
// Één volledige simulatie (rollout)
// ---------------------------------------------------------------------------
__device__
float gpu_simulate_one(SimState s, curandState* rng) {
    // Maximaal 300 stappen om oneindige loops te voorkomen
    for (int step = 0; step < 300 && !s.gameOver; step++) {
        int move = gpu_pickMove(&s, rng);
        gpu_applyMove(&s, move);
    }

    int numP = (int)s.numPlayers;
    if (numP <= 1) return 1.0f;

    if (s.gameOver) {
        // Score op basis van eindpositie (0=eerste, 1=tweede, ...)
        int myRank = (int)s.finishRank[s.myID];
        return (float)(numP - 1 - myRank) / (float)(numP - 1);
    }

    // Onvolledig spel: heuristiek op basis van kaartenaantal
    int myCards = gpu_totalCards(&s, (int)s.myID);
    int maxOpp  = 0;
    for (int i = 0; i < numP; i++) {
        if (i != (int)s.myID) {
            int c = gpu_totalCards(&s, i);
            if (c > maxOpp) maxOpp = c;
        }
    }
    float total = (float)(myCards + maxOpp + 1);
    return (float)maxOpp / total;
}

// ---------------------------------------------------------------------------
// Kernel: elke thread voert één simulatie uit
// ---------------------------------------------------------------------------
__global__
void kernel_batch_simulate(
    const SimState* __restrict__ states,
    float* __restrict__ results,
    int n,
    unsigned long long seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    curandState rng;
    curand_init(seed + (unsigned long long)idx, 0, 0, &rng);

    results[idx] = gpu_simulate_one(states[idx], &rng);
}

// ---------------------------------------------------------------------------
// Host-functies — geëxporteerd naar Go via DLL
// ---------------------------------------------------------------------------
extern "C" {

__declspec(dllexport) int __cdecl gpu_device_count() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) return 0;
    return count;
}

__declspec(dllexport) int __cdecl gpu_sim_state_size() {
    return (int)sizeof(SimState);
}

// Voert n simulaties parallel uit op de GPU.
// states[0..n-1]: invoerarrays (één SimState per simulatie)
// results[0..n-1]: uitvoerscores [0.0, 1.0]
// seed: willekeurig getal als basis voor curand
__declspec(dllexport) void __cdecl gpu_batch_simulate(
    const SimState* states,
    float*          results,
    int             n,
    unsigned long long seed
) {
    if (n <= 0) return;

    SimState* d_states = nullptr;
    float*    d_results = nullptr;

    size_t statesBytes  = (size_t)n * sizeof(SimState);
    size_t resultsBytes = (size_t)n * sizeof(float);

    cudaMalloc(&d_states,  statesBytes);
    cudaMalloc(&d_results, resultsBytes);

    cudaMemcpy(d_states, states, statesBytes, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize  = (n + blockSize - 1) / blockSize;
    kernel_batch_simulate<<<gridSize, blockSize>>>(d_states, d_results, n, seed);

    cudaMemcpy(results, d_results, resultsBytes, cudaMemcpyDeviceToHost);

    cudaFree(d_states);
    cudaFree(d_results);
}

} // extern "C"
