#define WIDTH  (100)
#define HEIGHT (100)
/////////////////////////////////////////////////////////////////////////////////////
//
// kernel for Schelling(Segregation) model
//
/////////////////////////////////////////////////////////////////////////////////////

#define NONE (-1)

void transHSVtoRGB(float *dst, const float h, const float s, const float v) {
    if (0.f == s) {
        dst[0] = dst[1] = dst[2] = v;
        return;
    }

    int Hi = (int)floor(h / 60.f) % 6;
    float f = h / 60.f - (float)Hi;
    float p = v * (1.f - s);
    float q = v * (1.f - s * f);
    float t = v * (1.f - (1.f - f) * s);

    if (Hi == 0) {
        dst[0] = v, dst[1] = t, dst[2] = p;
    } else if (Hi == 1) {
        dst[0] = q, dst[1] = v, dst[2] = p;
    } else if (Hi == 2) {
        dst[0] = p, dst[1] = v, dst[2] = t;
    } else if (Hi == 3) {
        dst[0] = p, dst[1] = q, dst[2] = v;
    } else if (Hi == 4) {
        dst[0] = t, dst[1] = p, dst[2] = v;
    } else {
        dst[0] = v, dst[1] = p, dst[2] = q;
    }
}

uint randomi(uint low, uint high, __global uint4* s) {
    const uint gid = get_global_id(0) + WIDTH * get_global_id(1);
    uint t;
    uint4 seed = s[gid];
    t = (seed.x^(seed.x<<11));
    seed.x = seed.y;
    seed.y = seed.z;
    seed.z = seed.w;
    seed.w = (seed.w^(seed.w>>19))^(t^(t>>8));
    s[gid] = seed;
    return low + ( seed.w % ( high - low ) );
}

float randomf(__global uint4* seed) {
    return ( randomi( 0, 10000000, seed ) / 10000000.0f );
}

int2 getTorus(int2 pos) {
    return (int2)( (pos.x + WIDTH) % WIDTH, (pos.y + HEIGHT) % HEIGHT );
}

int getOneDimIdx(int2 pos) {
    int2 _pos = getTorus(pos);
    return _pos.x + WIDTH * _pos.y;
}

__kernel
void writeColorObj(
        __global float4 *colorObj,
        __global int *status,
        __global float *colorValue)
{
    const int gid = getOneDimIdx( (int2)(get_global_id(0), get_global_id(1)) );
    int myStatus = status[gid];

    float rgb[3];
    rgb[0] = colorValue[3 * myStatus + 0];    
    rgb[1] = colorValue[3 * myStatus + 1];
    rgb[2] = colorValue[3 * myStatus + 2];

    colorObj[3 * gid + 0] = (float4)(rgb[0], rgb[1], rgb[2], rgb[0]);
    colorObj[3 * gid + 1] = (float4)(rgb[1], rgb[2], rgb[0], rgb[1]);
    colorObj[3 * gid + 2] = (float4)(rgb[2], rgb[0], rgb[1], rgb[2]);
}

__kernel
void updateCellStatus(
        __global uint4 *seed,
        __global int *status,
        __global int *newStatus
        )
{
    const int2 pos = (int2)(get_global_id(0), get_global_id(1));
    const int offset = getOneDimIdx(pos);

    int myStatus = status[offset];

    if (myStatus == 0) {
        if ( randomf(seed) < 0.1f ) {
            myStatus = 1;
        }
    }
    else if (myStatus == 1) {
        if ( randomf(seed) < 0.02f ) {
            myStatus = 2;
        }
    }
    else if (myStatus == 2) {
        if ( randomf(seed) < 0.02f ) {
            myStatus = 3;
        }
    }
    else if (myStatus == 3) {
        int top    = status[ getOneDimIdx( pos + (int2)(-1,  0) ) ];
        int bottom = status[ getOneDimIdx( pos + (int2)( 1,  0) ) ];
        int left   = status[ getOneDimIdx( pos + (int2)( 0, -1) ) ];
        int right  = status[ getOneDimIdx( pos + (int2)( 0,  1) ) ];
        if ( randomf(seed) < 0.0001f 
                || ((top == 4 || bottom == 4 || left == 4 || right == 4)
                    && randomf(seed) < 0.8f )) {
            myStatus = 4;
        } 
    }
    else if (myStatus == 4) {
        if ( randomf(seed) < 1.f ) {
            myStatus = 5;
        }
    }
    else if (myStatus == 5) {
        if ( randomf(seed) < 0.8f ) {
            myStatus = 6;
        }
    }
    else if (myStatus == 6) {
        if ( randomf(seed) < 0.8f ) {
            myStatus = 0;
        }
    }

    newStatus[offset] = myStatus;
}

__kernel
void changeStatus(
        __global int *status,
        __global int *newStatus
        )
{
    const int offset = getOneDimIdx( (int2)(get_global_id(0), get_global_id(1)) );
    status[offset] = newStatus[offset];
}

