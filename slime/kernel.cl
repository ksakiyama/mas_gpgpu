#define WIDTH  (100)
#define HEIGHT (100)
/////////////////////////////////////////////////////////////////////////////////////
//
// kernel for Slime model
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

float atomic_add_float(__global float* const address, const float value) {
  uint oldval, newval, readback;

  *(float*)&oldval = *address;
  *(float*)&newval = (*(float*)&oldval + value);
  while ((readback = atomic_cmpxchg((__global uint*)address, oldval, newval)) != oldval) {
    oldval = readback;
    *(float*)&newval = (*(float*)&oldval + value);
  }
  return *(float*)&oldval;
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

float2 getTorusFloat2(float2 pos) {
  float2 ret = pos;
  if( ret.x >= WIDTH  ) ret.x -= WIDTH;
  if( ret.y >= HEIGHT ) ret.y -= HEIGHT;
  if( ret.x < 0 ) ret.x += WIDTH;
  if( ret.y < 0 ) ret.y += HEIGHT;
  return ret;
}

int getOneDimIdx(int2 pos) {
  int2 _pos = getTorus(pos);
  return _pos.x + WIDTH * _pos.y;
}

float2 rotation(float2 coor, int angle) {
  float r = radians((float)angle);
  return (float2)( cos(r) * coor.x + -1.0f * sin(r) * coor.y,
                   sin(r) * coor.x + cos(r) * coor.y );
}

__kernel
  void writeAgentVertexObj(
  __global float4 *vertexObj,
  __global float2 *positions,
  __global int    *angles,
  const float patch)
{
  const int gid = get_global_id(0);

  float halfPatch = patch / 2;
  float2 pos = positions[gid];
  int angle = angles[gid];

  float2 fp[4] = {
    (float2)(     halfPatch     * 1.5f,              0 * 1.5f),
    (float2)(-1 * halfPatch     * 1.5f, -1 * halfPatch * 1.5f),
    (float2)(-1 * halfPatch / 2 * 1.5f,              0 * 1.5f),
    (float2)(-1 * halfPatch     * 1.5f,      halfPatch * 1.5f) };

  fp[0] = rotation( fp[0], angle );
  fp[1] = rotation( fp[1], angle );
  fp[2] = rotation( fp[2], angle );
  fp[3] = rotation( fp[3], angle );

  float2 centerPos = pos * patch + halfPatch;

  vertexObj[3 * gid + 0] = (float4)(centerPos.x + fp[0].x,
                                    centerPos.y + fp[0].y,
                                    0,
                                    centerPos.x + fp[1].x);
  vertexObj[3 * gid + 1] = (float4)(centerPos.y + fp[1].y,
                                    0,
                                    centerPos.x + fp[2].x,
                                    centerPos.y + fp[2].y);
  vertexObj[3 * gid + 2] = (float4)(0,
                                    centerPos.x + fp[3].x,
                                    centerPos.y + fp[3].y,
                                    0);
}

__kernel
  void writeSpaceColorObj(
  __global float4 *colorObj,
  __global float *chemical,
  const float hsv_h,
  const float min_value,
  const float max_value)
{
  const int gid = get_global_id(0) + WIDTH * get_global_id(1); 
  float hsv_s, hsv_v;
  float rgb[3];

  float max_min = fabs(max_value - min_value);
  float spaceParam = chemical[gid] - min_value;

  spaceParam = min(spaceParam, max_value);
  spaceParam = max(spaceParam, min_value);

  float d = 2.0f / max_min;

  hsv_s = 2.f - spaceParam * d;
  hsv_s = min(hsv_s, 1.0f);

  hsv_v = spaceParam * d;
  hsv_v = min(hsv_v, 1.0f);

  transHSVtoRGB(rgb, hsv_h, hsv_s, hsv_v);

  colorObj[3 * gid + 0] = (float4)(rgb[0], rgb[1], rgb[2], rgb[0]);
  colorObj[3 * gid + 1] = (float4)(rgb[1], rgb[2], rgb[0], rgb[1]);
  colorObj[3 * gid + 2] = (float4)(rgb[2], rgb[0], rgb[1], rgb[2]); 
}

__kernel
  void diffuse(
  __global float *dst,
  __global float *src,
  const float diffuseRate)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int gid = x + y * WIDTH;
  float value = 0.0f;
  for(int dy = -1; dy <= 1; dy++) {
    for(int dx = -1; dx <= 1; dx++) {
      if( dy == 0 && dx == 0 ) continue;
      int p = getOneDimIdx( (int2)(x + dx, y + dy) );
      //int p = ((x + dx + WIDTH) % WIDTH) + ((y + dy + HEIGHT) % HEIGHT) * WIDTH;
      value += src[p];
    }
  }

  dst[gid] = value / 8 * diffuseRate;
}

__kernel void moveIdealChemicalSpot(
  __global float2 *positions,
  __global int    *directions,
  __global float  *chemical,
  __global uint4  *seed,
  const    float  sniff_threshold)
{
  int gid = get_global_id(0);

  float2 posf;
  int2 posi;
  int pos1d, angle;

  posf = positions[gid];
  angle = directions[gid];

  posi = (int2)( round(posf.x), round(posf.y) );

  pos1d = getOneDimIdx(posi);

  if (chemical[pos1d] > sniff_threshold) {

    float ahead, myright, myleft;

    posi = (int2)( round( posf.x + cos( radians((float)angle) ) ), round( posf.y + sin( radians((float)angle) ) ) );
    //posi = getTorus(posi);
    pos1d = getOneDimIdx(posi);
    ahead = chemical[pos1d];

    posi = (int2)( round( posf.x + cos( radians((float)angle + 45) ) ), round( posf.y + sin( radians((float)angle + 45) ) ) );
    //posi = getTorus(posi);
    pos1d = getOneDimIdx(posi);
    myright = chemical[pos1d];

    posi = (int2)( round( posf.x + cos( radians((float)angle - 45) ) ), round( posf.y + sin( radians((float)angle - 45) ) ) );
    //posi = getTorus(posi);
    pos1d = getOneDimIdx(posi);
    myleft = chemical[pos1d];

    if( myright >= ahead || myleft >= ahead ) {
      if( myright >= myleft ) {
        angle += 45;
      } else {
        angle -= 45;
      }
    }
  }

  angle += randomi( 0, 41, seed ) - randomi( 0, 41, seed );

  angle = (angle + 360) % 360;

  posf += (float2)( cos( radians((float)angle) ), sin( radians((float)angle) ) ) ;

  posf = getTorusFloat2(posf);

  directions[gid] = angle;
  positions[gid] = posf;

  posi = (int2)( round(posf.x), round(posf.y) );
  //posi = getTorus(posi);
  pos1d = getOneDimIdx(posi);
  atomic_add_float( &chemical[pos1d], 2.0f );
}

