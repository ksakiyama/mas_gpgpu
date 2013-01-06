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
  fp[0] = (float2)(     halfPatch, -1 * halfPatch);
  fp[1] = (float2)(-1 * halfPatch, -1 * halfPatch);
  fp[2] = (float2)(-1 * halfPatch,      halfPatch);
  fp[3] = (float2)(     halfPatch,      halfPatch);

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
void writeSpaceColorObj()
{
}

__kernel
void moveToBestSugarSpot()
{
}

__kernel
void updateAgentParameter()
{
}

__kernel
void growupSpaceSugar()
{
}