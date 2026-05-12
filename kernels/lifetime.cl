// lifetime.cl

// ============================================================
// TODO-L1: update_lifetimes kernel
//
// Goal:
//   Decrease lifetime by dt and mark alive/dead:
//
//     remaining = life[i] - dt
//     alive[i] = 1 if remaining > 0 else 0
//
// Inputs:
//   - life        : float lifetime per particle
//   - dt          : time step
//   - numParticles: particle count
// Output:
//   - alive       : char (0 or 1) per particle
//
// Use a bounds check (id >= numParticles) to support padded NDRange.
// ============================================================
__kernel void update_lifetimes(__global const float* life,
                               __global char* alive,
                               float dt,
                               int numParticles)
{
    // Complete the codes]

    
}
