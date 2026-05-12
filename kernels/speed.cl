// speed.cl

/*------------------------------------------------------------------------
TODO # 1: compute_speed kernel
Goal: Compute speed = sqrt(vx^2 + vy^2 + vz^2) for each particle.
Input:
    - vel: float3 velocity vector per particle
    - numParticles: particle count
Output:
    - speed: float scalar per particle
Use bounds check id < numParticles to support padded NDRange.
------------------------------------------------------------------------*/






