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
  const uint gid = get_global_id(0);
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

int2 getTorus(int2 pos) {
  return (int2)( (pos.x + WIDTH) % WIDTH, (pos.y + HEIGHT) % HEIGHT );
}

int getOneDimIdx(int2 pos) {
  int2 _pos = getTorus(pos);
  return _pos.x + WIDTH * _pos.y;
}

__kernel
  void writeSpaceColorObj(
  __global float4 *colorObj,
  __global int *sugar,
  const float hsv_h,
  const float min_value,
  const float max_value,
  const uint flagInverse)
{
  const int gid = get_global_id(0) + WIDTH * get_global_id(1);
  float hsv_s, hsv_v;
  float rgb[3];

  float max_min = fabs(max_value - min_value);
  float spaceParam = (float)sugar[gid] - min_value;

  spaceParam = min(spaceParam, max_value);
  spaceParam = max(spaceParam, min_value);

  float d = 2.0f / max_min;

  hsv_s = 2.f - spaceParam * d;
  hsv_s = min(hsv_s, 1.0f);

  hsv_v = spaceParam * d;
  hsv_v = min(hsv_v, 1.0f);

  if (flagInverse) {
    float tmp = hsv_s;
    hsv_s = hsv_v;
    hsv_v = tmp;
  }

  transHSVtoRGB(rgb, hsv_h, hsv_s, hsv_v);

  colorObj[3 * gid + 0] = (float4)(rgb[0], rgb[1], rgb[2], rgb[0]);
  colorObj[3 * gid + 1] = (float4)(rgb[1], rgb[2], rgb[0], rgb[1]);
  colorObj[3 * gid + 2] = (float4)(rgb[2], rgb[0], rgb[1], rgb[2]);
}


