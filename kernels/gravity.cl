// gravity.cl

// Optional identity kernel for basic testing
__kernel void identity(__global const float3* inVel,
                       __global float3* outVel,
                       int numParticles)
{
    int id = get_global_id(0);
    if (id >= numParticles) return; // safety check
    outVel[id] = inVel[id];
}

// ============================================================
// TODO # 2: apply_gravity kernel
//
// Goal: Update velocity for each particle using:
//       v_new = v + g * dt
//
// Inputs:
//   - vel         : array of float3 velocities
//   - g           : constant float3 acceleration (e.g. gravity)
//   - dt          : time step
//   - numParticles: size of the simulation
// Output:
//   - newVel      : updated velocities
//
// IMPORTANT:
//   Because we may pad the global NDRange to test work-group sizes,
//   we MUST check id < numParticles before reading/writing.
//
// Steps for you:
//   1. Get id = get_global_id(0);
//   2. If id >= numParticles, return;
//   3. Read v = vel[id];
//   4. Compute v_new = v + g * dt;
//      v_new = (float3)(v.x + g.x * dt, v.y + g.y * dt, v.z + g.z * dt);
//   5. Write v_new into newVel[id]. newVel[id] = v_new
// ============================================================
__kernel void apply_gravity(__global const float3* vel,
                            __global float3* newVel,
                            float3 g,
                            float dt,
                            int numParticles)
{
    // Complete the codes
    



    
}
