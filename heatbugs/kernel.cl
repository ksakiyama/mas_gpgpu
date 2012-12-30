#define WIDTH  (100)
#define HEIGHT (100)
/////////////////////////////////////////////////////////////////////////////////////
//
// kernel for Schelling(Segregation) model
//
/////////////////////////////////////////////////////////////////////////////////////

#define NONE (-1)

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

float randomf(float h, __global uint4* seed) {
    return h * ( randomi( 0, 10000000, seed ) / 10000000.0f );
}

int2 getTorus(int2 pos) {
  return (int2)( (pos.x + WIDTH) % WIDTH, (pos.y + HEIGHT) % HEIGHT );
}

int getOneDimIdx(int2 pos) {
  int2 _pos = getTorus(pos);
  return _pos.x + WIDTH * _pos.y;
}

uint moveOldPosToNewPos(int2 oldPos, int2 newPos, __global int *space, int groupNum) {
  if ( NONE == atomic_cmpxchg( &space[ getOneDimIdx(newPos) ], NONE, groupNum ) ) {
    atomic_xchg( &space[ getOneDimIdx(oldPos) ], NONE );
    return 1;
  }
  return 0;
}

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

__kernel
  void writeAgentVertexObj(__global float4 *vertexObj,
  __global int2 *position,
  const float patch)
{
  const int agent_id = get_global_id(0);

  int2 pos_int = position[agent_id];
  float2 pos_float = (float2)( (float)pos_int.x, (float)pos_int.y );

  float2 fp[4];
  float halfPatch = patch / 2;
  fp[0] = (float2)(     halfPatch,              0);
  fp[1] = (float2)(             0, -1 * halfPatch);
  fp[2] = (float2)(-1 * halfPatch,              0);
  fp[3] = (float2)(            0,       halfPatch);

  float2 posOnScreen = pos_float * patch + halfPatch;

  vertexObj[3 * agent_id + 0] = (float4)(posOnScreen.x + fp[0].x,
    posOnScreen.y + fp[0].y,
    0,
    posOnScreen.x + fp[1].x);
  vertexObj[3 * agent_id + 1] = (float4)(posOnScreen.y + fp[1].y,
    0,
    posOnScreen.x + fp[2].x,
    posOnScreen.y + fp[2].y);
  vertexObj[3 * agent_id + 2] = (float4)(0,
    posOnScreen.x + fp[3].x,
    posOnScreen.y + fp[3].y,
    0);
}

__kernel
  void writeSpaceColorObj(__global float4 *colorObj,
  __global float *spaceTemp,
  const float hsv_h,
  const float min_value,
  const float max_value)
{
  const int gid = getOneDimIdx((int2)(get_global_id(0), get_global_id(1)));
  float temp = spaceTemp[gid];
  float hsv_s, hsv_v;
  float rgb[3];

  float max_min = fabs(max_value - min_value);
  temp = temp - min_value;

  temp = min(temp, max_value);
  temp = max(temp, min_value);

  float d = 2.0f / max_min;

  hsv_s = 2.f - temp * d;
  hsv_s = min(hsv_s, 1.0f);

  hsv_v = temp * d;
  hsv_v = min(hsv_v, 1.0f);

  transHSVtoRGB(rgb, hsv_h, hsv_s, hsv_v);

  colorObj[3 * gid + 0] = (float4)(rgb[0], rgb[1], rgb[2], rgb[0]);
  colorObj[3 * gid + 1] = (float4)(rgb[1], rgb[2], rgb[0], rgb[1]);
  colorObj[3 * gid + 2] = (float4)(rgb[2], rgb[0], rgb[1], rgb[2]);
}

__kernel
  void moveToIdealTempSpot(__global uint4 *seed,
  __global int   *space,
  __global int2  *bugPos,
  __global int   *bugIdealTemp,
  __global int   *bugOutputHeat,
  __global float *spaceTemp,
  const    float  randomMoveChance)
{
  const int gid = get_global_id(0);
  int2 pos = bugPos[gid];
  float idealTemp = (float)bugIdealTemp[gid];

  /* unhappinessを計算 */
  float unhappiness = fabs( idealTemp - spaceTemp[ getOneDimIdx(pos) ] );

  if (unhappiness == 0) {
    return; /* 何もしない */
  }

  /* ランダム移動 */
  if ( randomf( 1, seed ) < randomMoveChance ) {
    uint count = 0;
    while (count < 8) {
      int dx = randomi(0, 3, seed) - 1;
      int dy = randomi(0, 3, seed) - 1;

      if (dx == 0 && dy == 0) continue;

      int2 newPos = getTorus( pos + (int2)(dx, dy) );

      if ( moveOldPosToNewPos(pos, newPos, space, 0) ) {
        pos = newPos;
        break;
      }

      count++;
    }

  } else { /* 最適な場所を探す */
    /* 現在地の温度 */
    float hereTemp = spaceTemp[ getOneDimIdx(pos) ];

    /* 探索に使用するパラメータ */
    float bestTemp = hereTemp;
    int2 bestPos = pos;

    int flag = (hereTemp < idealTemp) ? 1 : 0;

    for (int dy = -1; dy <= 1; dy++) {
      for (int dx = -1; dx <= 1; dx++) {
        if (dx == 0 && dy == 0) continue;

        /* 目的地の場所、温度 */
        int2 newPos = pos + (int2)(dx, dy);
        float searchTemp = spaceTemp[ getOneDimIdx(newPos) ];

        if (flag) {/* より高温の場所へ */
          bestTemp = max(bestTemp, searchTemp);
        }
        else {/* より低温の場所へ */
          bestTemp = min(bestTemp, searchTemp);
        }

        if ( searchTemp == bestTemp ) {
          bestPos = newPos;
        }
      }
    }

    if (bestTemp != hereTemp) {
      if ( moveOldPosToNewPos(pos, bestPos, space, 0) ) {
        pos = bestPos;
      }
    }
  }

  bugPos[gid] = pos;
  spaceTemp[ getOneDimIdx(pos) ] += (float)bugOutputHeat[gid];
}

__kernel
  void diffuseSpaceTemp(__global float *dst,
  __global float *src,
  const float diffuseRate,
  const int range)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int gid = x + y * WIDTH;
  float value = 0.0f;
  for(int dy = -range; dy <= range; dy++) {
    for(int dx = -range; dx <= range; dx++) {
      if( dy == 0 && dx == 0 ) continue;
      int p = ((x + dx + WIDTH) % WIDTH) + ((y + dy + HEIGHT) % HEIGHT) * WIDTH;
      value += src[p];
    }
  }
  int length = 2 * range + 1;
  int cells = length * length - 1;
  dst[gid] = value / cells * diffuseRate;
}

