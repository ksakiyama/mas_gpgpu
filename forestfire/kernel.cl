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

uint moveOldPosToNewPos(int2 oldPos, int2 newPos, __global int *space, int groupNum) {
  if ( NONE == atomic_cmpxchg( &space[ getOneDimIdx(newPos) ], NONE, groupNum ) ) {
    atomic_xchg( &space[ getOneDimIdx(oldPos) ], NONE );
    return 1;
  }
  return 0;
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

__kernel
  void moveToBestSugarSpot(
  __global int2 *position,
  __global int  *vision,
  __global int  *space_sugar,
  __global int  *space,
  __global uint4 *seed
  )
{
  const int gid = get_global_id(0);
  const int2 pos = position[gid];
  const int v = vision[gid];

  int best_sugar = -1;
  uint best_distance = 100;

  int best_pos_count = 0;
  int2 array_best_pos[4];

  for (int y = -v; y <= v; y++) {
    int2 searchPos = pos + (int2)(0, y);
    int _space_sugar = space_sugar[ getOneDimIdx(searchPos) ];

    if (_space_sugar > best_sugar) {
      best_sugar = _space_sugar;
      best_distance = abs_diff(pos.y, searchPos.y);
      array_best_pos[0] = searchPos;
      best_pos_count = 1;
    }
    else if (_space_sugar == best_sugar) {
      uint d = abs_diff(pos.y, searchPos.y);

      if (d < best_distance) {
        best_distance = d;
        array_best_pos[0] = searchPos;
        best_pos_count = 1;
      }
      else if (d == best_distance) {
        array_best_pos[best_pos_count++] = searchPos;
      }
    }
  }

  for (int x = -v; x <= v; x++) {
    int2 searchPos = pos + (int2)(x, 0);
    int _space_sugar = space_sugar[ getOneDimIdx(searchPos) ];

    if (_space_sugar > best_sugar) {
      best_sugar = _space_sugar;
      best_distance = abs_diff(pos.x, searchPos.x);
      array_best_pos[0] = searchPos;
      best_pos_count = 1;
    }
    else if (_space_sugar == best_sugar) {
      uint d = abs_diff(pos.x, searchPos.x);

      if (d < best_distance) {
        best_distance = d;
        array_best_pos[0] = searchPos;
        best_pos_count = 1;
      }
      else if (d == best_distance) {
        array_best_pos[best_pos_count++] = searchPos;
      }
    }
  }

  int2 goalPos;

  if (best_pos_count == 1) {
    goalPos = array_best_pos[0];
  }
  else {
    int r = randomi(0, best_pos_count, seed);
    goalPos = array_best_pos[r];
  }

  if (moveOldPosToNewPos(pos, goalPos, space, 0)) {
    position[gid] = goalPos;
  }
}

__kernel
  void updateAgentParameter(
  __global int2 *position,
  __global int  *age,
  __global int  *max_age,
  __global int  *metabolism,
  __global int  *sugar,
  __global int  *death_flag,
  __global int  *space_sugar
  )
{
  const int gid = get_global_id(0);
  const int point = getOneDimIdx(position[gid]);

  int _age = age[gid];
  int _sugar = sugar[gid];

  _sugar += space_sugar[point];
  space_sugar[point] = 0;

  _sugar -= metabolism[gid];

  _age++;

  if (_age >= max_age[gid] || _sugar < 0) {
    death_flag[gid] = 1;
  }
  else {
    age[gid] = _age;
    sugar[gid] = _sugar;
  }
}

__kernel
  void growupSpaceSugar(
  __global int *sugar,
  __global int *max_sugar
  )
{
  const int2 xy = (int2)(get_global_id(0), get_global_id(1));
  const int point = getOneDimIdx(xy);

  if (sugar[point] < max_sugar[point]) {
    sugar[point] += 1;
  }
}

__kernel
  void generateNewAgent(
  __global int2  *position,
  __global int   *age,
  __global int   *max_age,
  __global int   *vision,
  __global int   *metabolism,
  __global int   *sugar,
  __global int   *death_flag,
  __global int   *space,
  __global uint4 *seed
  )
{
  const int gid = get_global_id(0);

  if (!death_flag[gid]) {
    return;
  }

  int2 pos = position[gid];

  while (1) {
    int _x = randomi(0, WIDTH,  seed);
    int _y = randomi(0, HEIGHT, seed);
    int2 newPos = getTorus(pos + (int2)(_x, _y));
    if (moveOldPosToNewPos(pos, newPos, space, 0)) {
      position[gid] = newPos;
      break;
    }
  }

  age[gid] = 0;
  max_age[gid] = randomi(60, 100 + 1, seed);
  vision[gid] = randomi(1, 6 + 1, seed);
  metabolism[gid] = randomi(1, 4 + 1, seed);
  sugar[gid] = randomi(5, 25 + 1, seed);
  death_flag[gid] = 0;
}
