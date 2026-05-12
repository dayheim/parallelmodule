// accel.cl

// Optional identity kernel
__kernel void identity(__global const float3* inForce,
                       __global float3* outForce,
                       int numParticles)
{
    int id = get_global_id(0);
    if (id >= numParticles) return;
    outForce[id] = inForce[id];
}

/*----------------------------------------------------------------
TODO #2: compute_accel kernel
Goal: Compute acceleration using a = F / m for each particle.
Inputs:
    - force: float3 force vector per particle
    - mass: float mass per particle
    - numParticles: total count
Output:
    - accel: float3 acceleration per particle
IMPORTANT:
    - Use id < numParticles check for padded NDRange.
    - Avoid division by zero mass (we guarantee mass > 0 on host).
------------------------------------------------------------------*/



