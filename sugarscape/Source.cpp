//#include "ColorHelper.h"
#include "SakiyaMas.h"

#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <iterator>
#include <map>

#if defined(_WIN32)||defined(_WIN64) /* Windows */
#include <GL/glew.h>
#include <GL/wglew.h>
#include <GL/freeglut.h>
#else
#include <GL/glew.h>
#include <GL/glxew.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/freeglut.h>
#endif

#include <CL/cl.h>
#include <CL/cl_gl.h>

class Sugarscape {
  const int window_width;
  const int window_height;
  const int width;
  const int height;
  const int population;

  GLuint agentVertexObj;
  GLuint agentColorObj;
  GLuint spaceVertexObj;
  GLuint spaceColorObj;

  const int empty;

  float patch;
  float halfPatch;

  // GLUT
  int mouse_button;
  int mouse_old_x;
  int mouse_old_y;
  float translate_x;
  float translate_y;
  float scale_size;
  bool pause;
  bool single;
  std::clock_t start_clock;
  std::clock_t end_clock;
  int fpsLimit;
  int fpsCount;
  int vsync;
  int ticks;

  // OpenCL
  cl_uint platformIdx;
  cl_uint deviceIdx;
  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_command_queue queue;
  cl_program program;

  cl_mem memSpace;
  cl_mem memSeed;
  cl_mem memPosition;
  cl_mem memAge;
  cl_mem memMaxAge;
  cl_mem memVision;
  cl_mem memSugar;
  cl_mem memMetabolism;
  cl_mem memDeathFlag;
  cl_mem memSpaceSugar;
  cl_mem memSpaceMaxSugar;
  cl_mem memAgentVertexObj;
  cl_mem memSpaceColorObj;

  std::map<std::string, cl_kernel> kernels;
  std::vector<cl_mem> buffers;

public:
  Sugarscape(int window_width_,
    int window_height_,
    int width_,
    int height_,
    int population_)
    : window_width(window_width_),
    window_height(window_height_),
    width(width_),
    height(height_),
    population(population_),
    agentVertexObj(0),
    agentColorObj(0),
    spaceVertexObj(0),
    spaceColorObj(0),
    empty(-1),
    mouse_button(0),
    translate_x(0),
    translate_y(0),
    scale_size(1),
    pause(false),
    single(false),
    fpsLimit(1),
    fpsCount(0),
    vsync(1),
    ticks(0),
    platformIdx(0),
    deviceIdx(0),
    context(0),
    queue(0),
    program(0)
  {
    patch = (float)window_width / width;
    halfPatch = patch / 2;
  }

  ~Sugarscape() {
    cleanup();
  }

  int init() {
    if (!initGlut()) {
      std::cerr << "Error: Initializing GLUT\n";
      return 0;
    }

    if (!initOpenCL()) {
      std::cerr << "Error: Initializing OpenCL\n";
      return 0;
    }

    std::vector<int> space(width * height, empty);
    std::vector<int> space_max_sugar(width * height, 0);

    if (!readMapFile(&space_max_sugar[0])) {
      std::cerr << "Error: Reading TXT File\n";
      return 0;
    }

    std::vector<int> pos(population * 2, 0);
    std::vector<int> age(population, 0);
    std::vector<int> max_age(population, 0);
    std::vector<int> vision(population, 0);
    std::vector<int> sugar(population, 0);
    std::vector<int> metabolism(population, 0);
    
    int agent_id = 0;
    while (agent_id < population) {
      int x = mcl::Random::random(width);
      int y = mcl::Random::random(height);
      if (space[getOneDimIdx(x, y)] == empty) {
        pos[2 * agent_id + 0] = x;
        pos[2 * agent_id + 1] = y;
        space[getOneDimIdx(x, y)] = agent_id;
        agent_id++;
      }
    }

    for (int i = 0; i < population; i++) {
      max_age[i] = mcl::Random::random(60, 101);
      vision[i] = mcl::Random::random(1, 7);
      metabolism[i] = mcl::Random::random(1, 5);
      sugar[i] = mcl::Random::random(5, 26);
    }

    std::vector<unsigned int> seed(population * 4, 0);
    for (int i = 0; i < population * 4; i++) {
      seed[i] = mcl::Random::random();
    }

    createAgentBufferObjs(&pos.front());
    createSpaceBufferObjs(&space_max_sugar.front());

    std::vector<int> deathFlag(population, 0);

    memSpace = createCLBuffer(&space.front(), space.size());
    memSeed = createCLBuffer(&seed.front(), seed.size());
    memPosition = createCLBuffer(&pos.front(), pos.size());
    memAge = createCLBuffer(&age.front(), age.size());
    memMaxAge = createCLBuffer(&max_age.front(), max_age.size());
    memVision = createCLBuffer(&vision.front(), vision.size());
    memSugar = createCLBuffer(&sugar.front(), sugar.size());
    memMetabolism = createCLBuffer(&metabolism.front(), metabolism.size());
    memDeathFlag = createCLBuffer(&deathFlag.front(), deathFlag.size());
    memSpaceSugar = createCLBuffer(&space_max_sugar.front(), space_max_sugar.size());
    memSpaceMaxSugar = createCLBuffer(&space_max_sugar.front(), space_max_sugar.size());

    memAgentVertexObj = createFromGLBuffer(agentVertexObj);
    memSpaceColorObj = createFromGLBuffer(spaceColorObj);

    for (size_t i = 0; i < buffers.size(); i++) {
      if (buffers[i] == 0) return 0;
    }

    int err = 1;
    err &= createKernel("writeAgentVertexObj");
    err &= createKernel("writeSpaceColorObj");
    err &= createKernel("moveToBestSugarSpot");
    err &= createKernel("updateAgentParameter");
    err &= createKernel("growupSpaceSugar");
    err &= createKernel("generateNewAgent");
    if (!err) return 0;

    if (!setArgs()) return 0;

    return 1;
  }

private:
  int run() {
    int err;
    err = moveToBestSugarSpot();
    if (err == 0) return 0;
    
    err = updateAgentParameter();
    if (err == 0) return 0;

    err = growupSpaceSugar();
    if (err == 0) return 0;

    err = generateNewAgent();
    if (err == 0) return 0;

    err = writeAgentVertexObj();
    if (err == 0) return 0;
    
    err = writeSpaceColorObj(mcl::Color::Yellow, 4, 0, -0.85f);
    if (err == 0) return 0;

    return 1;
  }

  int writeAgentVertexObj() {
    int err;
    err = acquireGLObject(memAgentVertexObj);
    if (err == 0) return 0;

    size_t gwSize = population;
    std::string krnl = "writeAgentVertexObj";
    err = launchKernel(krnl, &gwSize, 1);
    if (err == 0) return 0;

    err = releaseGLObject(memAgentVertexObj);
    if (err == 0) return 0;

    return 1;
  }

  int writeSpaceColorObj(float color_value, float min_value, float max_value, float bias) {
    std::string krnl = "writeSpaceColorObj";
    
    cl_uint flagInverse = (max_value < min_value);
    if (flagInverse) {
      float tmp = max_value;
      max_value = min_value;
      min_value = tmp;
    }

    if (bias != 0) {
      float max_min = fabs(max_value - min_value);
      if (bias < 0) {
        max_value += max_min * fabs(bias);
      } else {
        min_value -= max_min * bias;
      }
    }

    setArg(krnl, 3, min_value);
    setArg(krnl, 4, max_value);
    setArg(krnl, 5, flagInverse);

    int err;
    err = acquireGLObject(memSpaceColorObj);
    if (err == 0) return 0;

    size_t gwSize[2] = {width, height};
    err = launchKernel(krnl, gwSize, 2);
    if (err == 0) return 0;

    err = releaseGLObject(memSpaceColorObj);
    if (err == 0) return 0;

    return 1;
  }

  int moveToBestSugarSpot() {
    std::string krnl = "moveToBestSugarSpot";
    size_t gwSize = population;
    return launchKernel(krnl, &gwSize, 1);
  }

  int updateAgentParameter() {
    std::string krnl = "updateAgentParameter";
    size_t gwSize = population;
    return launchKernel(krnl, &gwSize, 1);
  }

  int growupSpaceSugar() {
    std::string krnl = "growupSpaceSugar";
    size_t gwSize[2] = {width, height};
    return launchKernel(krnl, gwSize, 2);
  }

  int generateNewAgent() {
    std::string krnl = "generateNewAgent";
    size_t gwSize = population;
    return launchKernel(krnl, &gwSize, 1);
  }

  int setArgs() {
    std::string krnl;
    int err, i;

    krnl = "writeAgentVertexObj";
    err = i = 0;
    err += setArg(krnl, i++, memAgentVertexObj);
    err += setArg(krnl, i++, memPosition);
    err += setArg(krnl, i++, patch);
    if (err != i) return 0;

    krnl = "writeSpaceColorObj";
    err = i = 0;
    err += setArg(krnl, i++, memSpaceColorObj);
    err += setArg(krnl, i++, memSpaceSugar);
    err += setArg(krnl, i++, mcl::Color::Yellow);
    if (err != i) return 0;

    krnl = "moveToBestSugarSpot";
    err = i = 0;
    err += setArg(krnl, i++, memPosition);
    err += setArg(krnl, i++, memVision);
    err += setArg(krnl, i++, memSpaceSugar);
    err += setArg(krnl, i++, memSpace);
    err += setArg(krnl, i++, memSeed);
    if (err != i) return 0;

    krnl = "updateAgentParameter";
    err = i = 0;
    err += setArg(krnl, i++, memPosition);
    err += setArg(krnl, i++, memAge);
    err += setArg(krnl, i++, memMaxAge);
    err += setArg(krnl, i++, memMetabolism);
    err += setArg(krnl, i++, memSugar);
    err += setArg(krnl, i++, memDeathFlag);
    err += setArg(krnl, i++, memSpaceSugar);
    if (err != i) return 0;

    krnl = "growupSpaceSugar";
    err = i = 0;
    err += setArg(krnl, i++, memSpaceSugar);
    err += setArg(krnl, i++, memSpaceMaxSugar);
    if (err != i) return 0;

    krnl = "generateNewAgent";
    err = i = 0;
    err += setArg(krnl, i++, memPosition);
    err += setArg(krnl, i++, memAge);
    err += setArg(krnl, i++, memMaxAge);
    err += setArg(krnl, i++, memVision);
    err += setArg(krnl, i++, memMetabolism);
    err += setArg(krnl, i++, memSugar);
    err += setArg(krnl, i++, memDeathFlag);
    err += setArg(krnl, i++, memSpace);
    err += setArg(krnl, i++, memSeed);
    if (err != i) return 0;

    return 1;
  }

  int initGlut() {
    int c = 1;
    char* dummy = {(char*)""};
    glutInit(&c, &dummy);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(window_width, window_height);
    glutInitWindowPosition(0, 0);
    glutCreateWindow("Schelling Model");
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);

    GLenum err = glewInit();
    if (err != GLEW_OK) {
      std::cerr << "Error: " << glewGetErrorString(err) << std::endl;
      return 0;
    }

    glViewport(0, 0, window_width, window_height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, window_width, window_height, 0);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);

    return 1;
  }

  int initOpenCL() {
    cl_int ret;

    cl_uint numPlatforms = 0;
    std::vector<cl_platform_id> platforms(10);
    ret = clGetPlatformIDs(10, &platforms[0], &numPlatforms);
    if (ret != CL_SUCCESS || numPlatforms == 0) {
      std::cerr << "Error " << ret << ": clGetPlatformIDs\n";
      return 0;
    }
    if (platformIdx > numPlatforms) platformIdx = 0;
    platform = platforms[platformIdx];

    cl_uint numDevices = 0;
    std::vector<cl_device_id> devices(10);
    ret = clGetDeviceIDs(platform,CL_DEVICE_TYPE_ALL, 10, &devices[0], &numDevices);
    if (ret != CL_SUCCESS || numDevices == 0) {
      std::cerr << "Error " << ret << ": clGetDeviceIDs\n";
      return 0;
    }
    if (deviceIdx > numDevices) deviceIdx = 0;
    device = devices[deviceIdx];

#if defined(_WIN32)||defined(_WIN64)
    cl_context_properties properties[] = {
      CL_CONTEXT_PLATFORM, (cl_context_properties)platform,
      CL_GL_CONTEXT_KHR,   (cl_context_properties)wglGetCurrentContext(),
      CL_WGL_HDC_KHR,      (cl_context_properties)wglGetCurrentDC(),
      0
    };
#else
    cl_context_properties properties[] = {
      CL_CONTEXT_PLATFORM, (cl_context_properties)platform,
      CL_GL_CONTEXT_KHR,   (cl_context_properties)glXGetCurrentContext(),
      CL_GLX_DISPLAY_KHR,  (cl_context_properties)glXGetCurrentDisplay(),
      0
    };
#endif
    context = clCreateContext(properties, 1, &device, NULL, NULL, &ret);
    if (ret != CL_SUCCESS) {
      std::cerr << "Error " << ret << ": clCreateContext\n";
      return 0;
    }
    queue = clCreateCommandQueue(context, device, 0, &ret);
    if (ret != CL_SUCCESS) {
      std::cerr << "Error " << ret << ": clCreateCommandQueue\n";
      return 0;
    }

    std::ifstream kernelFile("kernel.cl", std::ios::in);
    if (!kernelFile.is_open()) {
      std::cerr << "Error: Cannot open kernel.cl\n";
      return 0;
    }
    std::ostringstream oss;
    oss << kernelFile.rdbuf();
    std::string srcStdStr = oss.str();
    srcStdStr.erase(0, 50);

    std::stringstream ss;
    ss << "#define WIDTH (" << width << ")\n";
    ss << "#define HEIGHT (" << height << ")\n";

    std::string newSrcStdStr = ss.str() + srcStdStr;
    const char *srcStr = newSrcStdStr.c_str();
    program = clCreateProgramWithSource(context, 1, (const char**)&srcStr, NULL, &ret);
    if (ret != CL_SUCCESS) {
      std::cerr << "Error " << ret << ": clCreateProgramWithSource\n";
      return 0;
    }

    ret = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (ret != CL_SUCCESS) {
      std::cerr << "Error " << ret << ": clBuildProgram\n";
      return 0;
    }

    return 1;
  }

  void createAgentBufferObjs(int *agentPositions) {
    cl_float2 fp[4];
    fp[0].s[0] = halfPatch;      fp[0].s[1] = 0;
    fp[1].s[0] = 0;              fp[1].s[1] = -1 * halfPatch;
    fp[2].s[0] = -1 * halfPatch; fp[2].s[1] = 0;
    fp[3].s[0] = 0;              fp[3].s[1] = halfPatch;

    size_t size = sizeof(float) * 4 * 3 * population;
    std::vector<float> bufferObj(population * 4 * 3, 0);
    for (int i = 0; i < population; i++) {
      float x = agentPositions[2 * i + 0] * patch + halfPatch;
      float y = agentPositions[2 * i + 1] * patch + halfPatch;
      bufferObj[12 * i + 0 ] = x + fp[0].s[0];
      bufferObj[12 * i + 1 ] = y + fp[0].s[1];
      bufferObj[12 * i + 2 ] = 0;
      bufferObj[12 * i + 3 ] = x + fp[1].s[0];
      bufferObj[12 * i + 4 ] = y + fp[1].s[1];
      bufferObj[12 * i + 5 ] = 0;
      bufferObj[12 * i + 6 ] = x + fp[2].s[0];
      bufferObj[12 * i + 7 ] = y + fp[2].s[1];
      bufferObj[12 * i + 8 ] = 0;
      bufferObj[12 * i + 9 ] = x + fp[3].s[0];
      bufferObj[12 * i + 10] = y + fp[3].s[1];
      bufferObj[12 * i + 11] = 0;
    }

    glGenBuffers(1, &agentVertexObj);
    glBindBuffer(GL_ARRAY_BUFFER, agentVertexObj);
    glBufferData(GL_ARRAY_BUFFER, size, &bufferObj[0], GL_STATIC_DRAW);
    glFinish();

    float rgb[3];
    mcl::Color::trans(rgb, mcl::Color::Red, 0);

    for (int i = 0; i < population; i++) {
      bufferObj[12 * i + 0 ] = rgb[0]; // r
      bufferObj[12 * i + 1 ] = rgb[1]; // g
      bufferObj[12 * i + 2 ] = rgb[2]; // b
      bufferObj[12 * i + 3 ] = rgb[0];
      bufferObj[12 * i + 4 ] = rgb[1];
      bufferObj[12 * i + 5 ] = rgb[2];
      bufferObj[12 * i + 6 ] = rgb[0];
      bufferObj[12 * i + 7 ] = rgb[1];
      bufferObj[12 * i + 8 ] = rgb[2];
      bufferObj[12 * i + 9 ] = rgb[0];
      bufferObj[12 * i + 10] = rgb[1];
      bufferObj[12 * i + 11] = rgb[2];
    }

    glGenBuffers(1, &agentColorObj);
    glBindBuffer(GL_ARRAY_BUFFER, agentColorObj);
    glBufferData(GL_ARRAY_BUFFER, size, &bufferObj[0], GL_STATIC_DRAW);
    glFinish();
  }

  void createSpaceBufferObjs(int *spaceSugar) {
    size_t size = sizeof(float) * 4 * 3 * width * height;
    std::vector<float> bufferObj(width * height * 4 * 3, 0);
    for (int i = 0; i < width * height; i++) {
      int x = i % width;
      int y = i / width;
      bufferObj[12 * i + 0 ] = x * patch;
      bufferObj[12 * i + 1 ] = y * patch;
      bufferObj[12 * i + 2 ] = 0;
      bufferObj[12 * i + 3 ] = x * patch + patch;
      bufferObj[12 * i + 4 ] = y * patch;
      bufferObj[12 * i + 5 ] = 0;
      bufferObj[12 * i + 6 ] = x * patch + patch;
      bufferObj[12 * i + 7 ] = y * patch + patch;
      bufferObj[12 * i + 8 ] = 0;
      bufferObj[12 * i + 9 ] = x * patch;
      bufferObj[12 * i + 10] = y * patch + patch;
      bufferObj[12 * i + 11] = 0;
    }

    glGenBuffers(1, &spaceVertexObj);
    glBindBuffer(GL_ARRAY_BUFFER, spaceVertexObj);
    glBufferData(GL_ARRAY_BUFFER, size, &bufferObj[0], GL_STATIC_DRAW);
    glFinish();

    for (int i = 0; i < width * height; i++) {
      float rgb[3];
      float bias = 1;
      if (spaceSugar[i] == 1) bias = 0.8f;
      else if (spaceSugar[i] == 2) bias = 0.5f;
      else if (spaceSugar[i] == 3) bias = 0.25f;
      else if (spaceSugar[i] == 4) bias = -0.05f;
      mcl::Color::trans(rgb, mcl::Color::Yellow, bias);
      bufferObj[12 * i + 0 ] = rgb[0]; // r
      bufferObj[12 * i + 1 ] = rgb[1]; // g
      bufferObj[12 * i + 2 ] = rgb[2]; // b
      bufferObj[12 * i + 3 ] = rgb[0];
      bufferObj[12 * i + 4 ] = rgb[1];
      bufferObj[12 * i + 5 ] = rgb[2];
      bufferObj[12 * i + 6 ] = rgb[0];
      bufferObj[12 * i + 7 ] = rgb[1];
      bufferObj[12 * i + 8 ] = rgb[2];
      bufferObj[12 * i + 9 ] = rgb[0];
      bufferObj[12 * i + 10] = rgb[1];
      bufferObj[12 * i + 11] = rgb[2];
    }

    glGenBuffers(1, &spaceColorObj);
    glBindBuffer(GL_ARRAY_BUFFER, spaceColorObj);
    glBufferData(GL_ARRAY_BUFFER, size, &bufferObj[0], GL_STATIC_DRAW);
    glFinish();
  }

  int readMapFile(int *sugarPtr) {
    std::ifstream ifs;

    std::string filename = "sugar-map.txt";

    if ((width == 50 && height == 50) || (width == 1000 && height == 1000)) {
      if (width == 1000 && height == 1000) {
        filename = "large-sugar-map.txt";
      }
    }
    else {
      std::cerr << "Invalid space size.\n";
      return 0;
    }

    ifs.open(filename.c_str());
    if (!ifs.is_open()) {
      std::cerr << "Failed to open file.\n";
      return 0;
    }

    /* Copy */
    std::vector<char> map_data;
    std::copy(std::istream_iterator<char>(ifs), std::istream_iterator<char>(), std::back_inserter(map_data));

    /* Convert int data */
    std::vector<int> sugar;
    for (size_t i = 0; i < map_data.size(); i++) {
      sugar.push_back( (int)(map_data[i] - '0') );
    }

    /* Set SpaceData<T> */
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        sugarPtr[ getOneDimIdx(x, y) ] = sugar[ getOneDimIdx(x, y) ];
      }
    }

    return 1;
  }

  template <class T>
  int setArg(std::string kernelName, cl_uint num, T value) {
    cl_int ret = clSetKernelArg(
      kernels[kernelName],
      num,
      details::SetArgHandler<T>::size(value),//sizeof(T),
      details::SetArgHandler<T>::ptr(value));

    if (ret != CL_SUCCESS) {
      std::cerr << "Error " << ret << ": clSetKernelArg\n";
      return 0;
    }

    return 1;
  }

  int waitForEvent(cl_event *event_) {
    cl_int ret = clWaitForEvents(1, event_);
    if (ret != CL_SUCCESS) {
      std::cerr << "Error " << ret << ": clWaitForEvents\n";
      return 0;
    }
    clReleaseEvent(*event_);
    return 1;
  }

  int launchKernel(std::string kernelName, size_t *gwSize, cl_uint dim) {
    cl_event event_;
    cl_int ret = clEnqueueNDRangeKernel(queue,
      kernels[kernelName],
      dim,
      NULL,
      gwSize,
      NULL,
      0,
      NULL,
      &event_);
    if (ret != CL_SUCCESS) {
      std::cerr << "Error " << ret << ": clEnqueueNDRangeKernel\n";
      return 0;
    }

    return waitForEvent(&event_);
  }

  int acquireGLObject(cl_mem memSrc) {
    cl_event event_;
    cl_int ret = clEnqueueAcquireGLObjects(queue,
      1, &memSrc,
      0,
      NULL,
      &event_);
    if (ret != CL_SUCCESS) {
      std::cerr << "Error " << ret << ": clEnqueueAcquireGLObjects\n";
      return 0;
    }

    return waitForEvent(&event_);
  }

  int releaseGLObject(cl_mem memSrc) {
    cl_event event_;
    cl_int ret = clEnqueueReleaseGLObjects(queue,
      1,
      &memSrc,
      0,
      NULL,
      &event_);
    if (ret != CL_SUCCESS) {
      std::cerr << "Error " << ret << ": clEnqueueReleaseGLObjects\n";
      return 0;
    }

    return waitForEvent(&event_);
  }

  template <class T>
  cl_mem createCLBuffer(T *hostPtr, size_t num_element) {
    cl_int ret;
    cl_mem memObj = clCreateBuffer(context,
      CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
      sizeof(T) * num_element,
      hostPtr,
      &ret);
    if (ret != CL_SUCCESS) {
      std::cerr << "Error " << ret << ": clCreateBuffer\n";
      return 0;
    }
    buffers.push_back(memObj);
    return memObj;
  }

  cl_mem createFromGLBuffer(GLuint glBufferObj) {
    cl_int ret;
    cl_mem memObj = clCreateFromGLBuffer(context,
      CL_MEM_WRITE_ONLY,
      glBufferObj,
      &ret);
    if (ret != CL_SUCCESS) {
      std::cerr << "Error " << ret << ": clCreateFromGLBuffer\n";
      return 0;
    }
    buffers.push_back(memObj);
    return memObj;
  }

  int createKernel(std::string krnlName) {
    cl_int ret;
    kernels.insert( std::map<std::string, cl_kernel>::value_type( krnlName, (cl_kernel)0) );
    kernels[krnlName] = clCreateKernel(program, krnlName.c_str(), &ret);
    if (ret != CL_SUCCESS) {
      std::cerr << "Error " << ret << ": clCreateKernel\n";
      return 0;
    }
    return 1;
  }

  int getOneDimIdx(int x, int y) {
    return x + width * y;
  }

  /* GLUT Callback */
public:
  void setPlatformIndex(cl_uint idx) {
    platformIdx = idx;
  }

  void setDeviceIndex(cl_uint idx) {
    deviceIdx = idx;
  }

  void setVsync(bool arg) {
    vsync = arg ? 1 : 0;
  }

  void setStop(bool arg) {
    pause = arg ? true : false;
  }

  static void _wrapDisplay() { Sugarscape* ptr = *(get_ptr()); ptr->display(); }
  static void _wrapIdle() { Sugarscape* ptr = *(get_ptr()); ptr->idle(); }
  static void _wrapKeyboard(unsigned char key, int x, int y) { Sugarscape* ptr = *(get_ptr()); ptr->keyboard(key, x, y); }
  static void _wrapMouse(int button, int state, int x, int y) { Sugarscape* ptr = *(get_ptr()); ptr->mouse(button, state, x, y); }
  static void _wrapMotion(int x, int y) { Sugarscape* ptr = *(get_ptr()); ptr->motion(x, y); }

  void start() {
#if defined(_WIN32)||(_WIN64)
    if (glewGetExtension("WGL_EXT_swap_control")) {
      wglSwapIntervalEXT(vsync);
    }
#endif
    Sugarscape** ss_ptr = get_ptr();
    *ss_ptr = this;
    glutDisplayFunc(_wrapDisplay);
    glutKeyboardFunc(_wrapKeyboard);
    glutIdleFunc(_wrapIdle);
    glutMouseFunc(_wrapMouse);
    glutMotionFunc(_wrapMotion);

    start_clock = std::clock();
    glutMainLoop();
  }

private:
  static Sugarscape** get_ptr() {
    static Sugarscape* ptr;
    return &ptr;
  }

  void display() {
    glClear(GL_COLOR_BUFFER_BIT);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(translate_x, translate_y, 0.0);
    glScalef(scale_size, scale_size, 1.0);

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    glBindBuffer(GL_ARRAY_BUFFER, spaceVertexObj);
    glVertexPointer(3, GL_FLOAT, 0, 0);

    glBindBuffer(GL_ARRAY_BUFFER, spaceColorObj);
    glColorPointer(3, GL_FLOAT, 0, 0);

    glDrawArrays(GL_QUADS, 0, width * height * 4);

    glBindBuffer(GL_ARRAY_BUFFER, agentVertexObj);
    glVertexPointer(3, GL_FLOAT, 0, 0);

    glBindBuffer(GL_ARRAY_BUFFER, agentColorObj);
    glColorPointer(3, GL_FLOAT, 0, 0);

    glDrawArrays(GL_QUADS, 0, population * 4);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);

    glutSwapBuffers();
  }

  void idle() {
    if (!pause || single) {
      int err = run();
      if (err == 0) {
        glutLeaveMainLoop();
      }
      ticks++;
      single = false;
    }
    glutPostRedisplay();
    calcFPS();
  }

  void keyboard(unsigned char key, int x, int y) {
    switch (key) {
    case 27:
      glutLeaveMainLoop();
      break;
    case 'r':
      translate_x = 0;
      translate_y = 0;
      scale_size = 1.0;
      break;
    case 's':
      pause = !pause;
      start_clock = std::clock();
      fpsLimit = 5;
      fpsCount = 0;
      break;
    case 'd':
      pause = true;
      single = true;
      start_clock = std::clock();
      fpsLimit = 1;
      fpsCount = 0;
      break;
#if defined(_WIN32)||defined(_WIN64)
    case 'v':
      if(glewGetExtension("WGL_EXT_swap_control")) {
        vsync = 1 - vsync;
        wglSwapIntervalEXT(vsync);
      }
      break;
#endif
    }
  }

  void mouse(int button, int state, int x, int y) {
    if (state == GLUT_DOWN) {
      mouse_button |= 1 << button;
    } else if (state == GLUT_UP) {
      mouse_button = 0;
    }

    static unsigned int countMouseWheel = 0;
    if (button == 3 || button == 4) {
      countMouseWheel++;
      if (countMouseWheel % 2 == 0) return;

      if (button == 3) {
        scale_size += 0.5f;
      } else if (button == 4) {
        scale_size -= 0.5f;
      }

      if(scale_size < 1) scale_size = 1.0;

      if(translate_x < -window_width * (scale_size - 1)) {
        translate_x = -window_width * (scale_size - 1);
      }

      if(translate_y < -window_height * (scale_size - 1)) {
        translate_y = -window_height * (scale_size - 1);
      }
    }

    mouse_old_x = x;
    mouse_old_y = y;
  }

  void motion(int x, int y) {
    float dx, dy;
    dx = (float)(x - mouse_old_x);
    dy = (float)(y - mouse_old_y);

    if (mouse_button & 1) {	// left click
      translate_x += dx;

      if (translate_x < -window_width * (scale_size - 1)) {
        translate_x = -window_width * (scale_size - 1);
      } else if (translate_x >= 0) {
        translate_x = 0;
      }

      translate_y += dy;

      if (translate_y < -window_height * (scale_size - 1)) {
        translate_y = -window_height * (scale_size - 1);
      } else if (translate_y >= 0) {
        translate_y = 0;
      }
    }

    mouse_old_x = x;
    mouse_old_y = y;
  }

  void calcFPS() {
    double ifps;
    fpsCount++;
    if (fpsCount >= fpsLimit) {
      end_clock = clock();

      char tmp[256];
      float milliseconds = ((float)end_clock - (float)start_clock) / CLOCKS_PER_SEC * 1000;
      milliseconds /= (float)fpsCount;

      ifps = 1.f / (milliseconds / 1000.f);
      std::sprintf(tmp, "MASCL(%d * %d, %d agents) : %0.1f FPS | %d ticks | Scale %.3f",
        window_width, window_height, population, ifps, ticks, scale_size);

      glutSetWindowTitle(tmp);
      fpsCount = 0;
      fpsLimit = (ifps > 1.f) ? (int)ifps : 1;

      start_clock = clock();
    }
  }

  void cleanup() {
    if (agentVertexObj != 0) {
      glBindBuffer(1, agentVertexObj);
      glDeleteBuffers(1, &agentVertexObj);
    }
    if (agentColorObj != 0) {
      glBindBuffer(1, agentColorObj);
      glDeleteBuffers(1, &agentColorObj);
    }
    if (spaceVertexObj != 0) {
      glBindBuffer(1, spaceVertexObj);
      glDeleteBuffers(1, &spaceVertexObj);
    }
    if (spaceColorObj != 0) {
      glBindBuffer(1, spaceColorObj);
      glDeleteBuffers(1, &spaceColorObj);
    }
    if (!buffers.empty()) {
      for (size_t i = 0; i < buffers.size(); i++) {
        if (buffers[i] != 0) clReleaseMemObject(buffers[i]);
      }
    }
    if (!kernels.empty()) {
      std::map<std::string, cl_kernel>::iterator i;
      for (i = kernels.begin(); i != kernels.end(); i++) {
        if ((*i).second != 0) clReleaseKernel((*i).second);
      }
    }
    if (program != 0) clReleaseProgram(program);
    if (queue != 0) clReleaseCommandQueue(queue);
    if (context != 0) clReleaseContext(context);
  }
};

int main(int argc, char* argv[]) {
  const int window_width  = 1000;
  const int window_height = 1000;

  //const int width  = 50;
  //const int height = 50;
  //const int num_agents = 50;

  const int width  = 1000;
  const int height = 1000;
  const int num_agents = 50000;

  Sugarscape sugarscape(
    window_width,
    window_height,
    width,
    height,
    num_agents
    );

  if (!sugarscape.init()) {
    return 1;
  }

  //sugarscape.setVsync(0);
  sugarscape.setStop(true);
  sugarscape.start();

  std::cout << "Simulation finished\n";

  return 0;
}
