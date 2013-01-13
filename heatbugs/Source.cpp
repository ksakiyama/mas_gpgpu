#include "../common/SakiyaMas.h"

#include <iostream>
#include <vector>
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>

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

class Heatbugs {
  const int window_width;
  const int window_height;
  const int width;
  const int height;
  const int population;

  const int max_ideal_temp;
  const int min_ideal_temp;
  const int max_output_heat;
  const int min_output_heat;
  const float randomMoveProb;
  const float evaporateRate;

  GLuint agent_vertex_vbo;
  GLuint agent_color_vbo;
  GLuint space_vertex_vbo;
  GLuint space_color_vbo;

  const int empty;

  float patch;
  float halfPatch;

  std::vector<int> space;
  std::vector<int> bugPosition;
  std::vector<int> bugOutputHeat;
  std::vector<int> bugIdealTemp;

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

  cl_kernel moveToIdealTempSpot;
  cl_kernel diffuseSpaceTemp;
  cl_kernel writeAgentVertexObj;
  cl_kernel writeSpaceColorObj;

  cl_mem memSpace;
  cl_mem memSeed;
  cl_mem memBugPosition;
  cl_mem memBugOutputHeat;
  cl_mem memBugIdealTemp;
  cl_mem memSpaceTemp1;
  cl_mem memSpaceTemp2;
  cl_mem memVertexObj;
  cl_mem memColorObj;
  std::vector<cl_mem> buffers;

public:
  Heatbugs(int window_width_,
    int window_height_,
    int width_,
    int height_,
    int population_)
    : max_ideal_temp(31000),
    min_ideal_temp(17000),
    max_output_heat(10000),
    min_output_heat(6000),
    randomMoveProb(0.1f),
    evaporateRate(0.99f),
    window_width(window_width_),
    window_height(window_height_),
    width(width_),
    height(height_),
    population(population_),
    agent_vertex_vbo(0),
    agent_color_vbo(0),
    space_vertex_vbo(0),
    space_color_vbo(0),
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
    program(0),
    moveToIdealTempSpot(0),
    diffuseSpaceTemp(0),
    writeAgentVertexObj(0),
    writeSpaceColorObj(0),
    memSpace(0),
    memSeed(0),
    memBugPosition(0),
    memBugOutputHeat(0),
    memBugIdealTemp(0),
    memSpaceTemp1(0),
    memSpaceTemp2(0),
    memVertexObj(0),
    memColorObj(0)
  {
    if (!initGlut()) {
      std::cerr << "Error: GLUT Initialization\n";
      return;
    }

    if (!initOpenCL()) {
      std::cerr << "Error: OpenCL Initialization\n";
      return;
    }

    patch = (float)window_width / width;
    halfPatch = patch / 2;

    space = std::vector<int>(width * height, empty);

    bugPosition = std::vector<int>(population * 2, 0);

    int agent_id = 0;
    while (agent_id < population) {
      int x = mcl::Random::random(width);
      int y = mcl::Random::random(height);
      if (space[ getOneDimIdx(x, y) ] == empty) {
        space[ getOneDimIdx(x, y) ] = agent_id;
        bugPosition[2 * agent_id + 0] = x;
        bugPosition[2 * agent_id + 1] = y;
        agent_id++;
      }
    }

    bugOutputHeat = std::vector<int>(population, 0);
    bugIdealTemp  = std::vector<int>(population, 0);
    for (int agent_id = 0; agent_id < population; agent_id++) {
      bugOutputHeat[agent_id] = mcl::Random::random(min_output_heat, max_output_heat + 1);
      bugIdealTemp[agent_id] = mcl::Random::random(min_ideal_temp, max_ideal_temp + 1);
    }

    createVertexObjs();
    createColorObjs();

    if (!createCLBuffers()) {
      return;
    }

    if (!setKernelArgs()) {
      return;
    }

    space.clear();
    bugPosition.clear();
    bugOutputHeat.clear();
    bugIdealTemp.clear();

    start();
  }

  ~Heatbugs() {
    if (agent_vertex_vbo != 0) {
      glBindBuffer(1, agent_vertex_vbo);
      glDeleteBuffers(1, &agent_vertex_vbo);
    }
    if (agent_color_vbo != 0) {
      glBindBuffer(1, agent_color_vbo);
      glDeleteBuffers(1, &agent_color_vbo);
    }
    if (space_vertex_vbo != 0) {
      glBindBuffer(1, space_vertex_vbo);
      glDeleteBuffers(1, &space_vertex_vbo);
    }
    if (space_color_vbo != 0) {
      glBindBuffer(1, space_color_vbo);
      glDeleteBuffers(1, &space_color_vbo);
    }
    if (!buffers.empty()) {
      for (size_t i = 0; i < buffers.size(); i++) {
        if (buffers[i] != 0) clReleaseMemObject(buffers[i]);
      }
    }
    if (moveToIdealTempSpot != 0) clReleaseKernel(moveToIdealTempSpot);
    if (diffuseSpaceTemp != 0) clReleaseKernel(diffuseSpaceTemp);
    if (writeAgentVertexObj != 0) clReleaseKernel(writeAgentVertexObj);
    if (writeSpaceColorObj != 0) clReleaseKernel(writeSpaceColorObj);
    if (program != 0) clReleaseProgram(program);
    if (queue != 0) clReleaseCommandQueue(queue);
    if (context != 0) clReleaseContext(context);
  }

  void start() {
#if defined(_WIN32)||(_WIN64)
    if (glewGetExtension("WGL_EXT_swap_control")) {
      wglSwapIntervalEXT(vsync);
    }
#endif
    Heatbugs** heat_ptr = get_ptr();
    *heat_ptr = this;
    glutDisplayFunc(_wrapDisplay);
    glutKeyboardFunc(_wrapKeyboard);
    glutIdleFunc(_wrapIdle);
    glutMouseFunc(_wrapMouse);
    glutMotionFunc(_wrapMotion);

    start_clock = std::clock();
    glutMainLoop();
  }

  void setVsync(bool arg) {
    if (arg) vsync = 1;
    else vsync = 0;
  }

  void setStop(bool arg) {
    if (arg) pause = true;
  }

private:
  int run() {
    int err;
    err = move(memSpaceTemp1);
    if (err == 0) return 0;

    err = diffuse(memSpaceTemp2, memSpaceTemp1);
    if (err == 0) return 0;

    err = updateAgentPosition();
    if (err == 0) return 0;

    err = scaleSpaceColor(memSpaceTemp2, mcl::Color::Red, 0, 32000, -0.5f);
    if (err == 0) return 0;

    //exchange
    cl_mem memTmp = memSpaceTemp1;
    memSpaceTemp1 = memSpaceTemp2;
    memSpaceTemp2 = memTmp;

    return 1;
  }

  int move(cl_mem temp) {
    size_t gwSize = population;

    int err = 0;
    err = setArg(&moveToIdealTempSpot, 5, temp);
    if (err == 0) return 0;

    err = launchKernel(&moveToIdealTempSpot, &gwSize, 1);
    if (err == 0) return 0;

    return 1;
  }

  int diffuse(cl_mem tempDst, cl_mem tempSrc) {
    size_t gwSize[2] = {width, height};

    int err = 0;
    int i = 0;
    err += setArg(&diffuseSpaceTemp, i++, tempDst);
    err += setArg(&diffuseSpaceTemp, i++, tempSrc);
    err += setArg(&diffuseSpaceTemp, i++, evaporateRate);
    err += setArg(&diffuseSpaceTemp, i++, 1);
    if (err != i) {
      printf("%d, %d\n", err, i);
      puts("a");
      return 0;
    }

    err = launchKernel(&diffuseSpaceTemp, gwSize, 2);
    if (err == 0) {
      puts("b");
      return 0;
    }
    return 1;
  }

  int updateAgentPosition() {
    size_t gwSize = population;

    int err;
    err = acquireGLObject(memVertexObj);
    if (err == 0) return 0;

    err = launchKernel(&writeAgentVertexObj, &gwSize, 1);
    if (err == 0) return 0;

    err = releaseGLObject(memVertexObj);
    if (err == 0) return 0;

    return 1;
  }

  int scaleSpaceColor(cl_mem temp, float color_value, float min_value, float max_value, float bias = 0) {
    size_t gwSize[2] = {width, height};

    if (bias != 0) {
        float max_min = fabs(max_value - min_value);
        if (bias < 0) {
            max_value += max_min * fabs(bias);
        } else {
            min_value -= max_min * bias;
        }
    }

    cl_int ret = CL_SUCCESS;
    ret |= clSetKernelArg(writeSpaceColorObj, 1, sizeof(cl_mem), &temp);
    ret |= clSetKernelArg(writeSpaceColorObj, 2, sizeof(float), &color_value);
    ret |= clSetKernelArg(writeSpaceColorObj, 3, sizeof(float), &min_value);
    ret |= clSetKernelArg(writeSpaceColorObj, 4, sizeof(float), &max_value);
    if (ret != CL_SUCCESS) return 0;

    int err;
    err = acquireGLObject(memColorObj);
    if (err == 0) return 0;

    err = launchKernel(&writeSpaceColorObj, gwSize, 2);
    if (err == 0) return 0;

    err = releaseGLObject(memColorObj);
    if (err == 0) return 0;

    return 1;
  }

  template <class T>
  int setArg(cl_kernel *krnl, cl_uint num, T value) {
    cl_int ret = clSetKernelArg(
      *krnl,
      num,
      details::SetArgHandler<T>::size(value),//sizeof(T),
      details::SetArgHandler<T>::ptr(value));

    if (ret != CL_SUCCESS) {
      return 0;
    }

    return 1;
  }

  int waitForEvent(cl_event *event_) {
    cl_int ret = clWaitForEvents(1, event_);
    if (ret != CL_SUCCESS) return 0;
    clReleaseEvent(*event_);
    return 1;
  }

  int launchKernel(cl_kernel *krnl, size_t *gwSize, cl_uint dim) {
    cl_event event_;
    cl_int ret = clEnqueueNDRangeKernel(queue,
      *krnl,
      dim,
      NULL,
      gwSize,
      NULL,
      0,
      NULL,
      &event_);
    if (ret != CL_SUCCESS) return 0;

    return waitForEvent(&event_);
  }

  int acquireGLObject(cl_mem memSrc) {
    cl_event event_;
    cl_int ret = clEnqueueAcquireGLObjects(queue,
      1, &memSrc,
      0,
      NULL,
      &event_);
    if (ret != CL_SUCCESS) return 0;

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
    if (ret != CL_SUCCESS) return 0;

    return waitForEvent(&event_);
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
    if (ret != CL_SUCCESS || numPlatforms == 0) return 0;
    platform = platforms[platformIdx];

    cl_uint numDevices = 0;
    std::vector<cl_device_id> devices(10);
    ret = clGetDeviceIDs(platform,CL_DEVICE_TYPE_ALL, 10, &devices[0], &numDevices);
    if (ret != CL_SUCCESS || numDevices == 0) return 0;
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
    if (ret != CL_SUCCESS) return 0;
    queue = clCreateCommandQueue(context, device, 0, &ret);
    if (ret != CL_SUCCESS) return 0;

    std::ifstream kernelFile("kernel.cl", std::ios::in);
    if (!kernelFile.is_open()) return 0;

    std::ostringstream oss;
    oss << kernelFile.rdbuf();
    std::string srcStdStr = oss.str();
    srcStdStr.erase(0, 50);

    std::stringstream ss;
    ss << "#define WIDTH (" << width << ")\n";
    ss << "#define HEIGHT (" << height << ")\n";
    //ss << "#define PATCH (" << patch << ")\n";

    std::string newSrcStdStr = ss.str() + srcStdStr;
    const char *srcStr = newSrcStdStr.c_str();
    program = clCreateProgramWithSource(context, 1, (const char**)&srcStr, NULL, &ret);
    if (ret != CL_SUCCESS) return 0;

    ret = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (ret != CL_SUCCESS) return 0;

    moveToIdealTempSpot = clCreateKernel(program, "moveToIdealTempSpot", &ret);
    if (ret != CL_SUCCESS) return 0;
    diffuseSpaceTemp = clCreateKernel(program, "diffuseSpaceTemp", &ret);
    if (ret != CL_SUCCESS) return 0;
    writeAgentVertexObj = clCreateKernel(program, "writeAgentVertexObj", &ret);
    if (ret != CL_SUCCESS) return 0;
    writeSpaceColorObj = clCreateKernel(program, "writeSpaceColorObj", &ret);
    if (ret != CL_SUCCESS) return 0;

    return 1;
  }

  void createVertexObjs() {
    cl_float2 fp[4];
    fp[0].s[0] = halfPatch;      fp[0].s[1] = 0;
    fp[1].s[0] = 0;              fp[1].s[1] = -1 * halfPatch;
    fp[2].s[0] = -1 * halfPatch; fp[2].s[1] = 0;
    fp[3].s[0] = 0;              fp[3].s[1] = halfPatch;

    size_t agent_vbo_size = sizeof(float) * 3 * 4 * population;
    std::vector<float> agent_vertex(population * 3 * 4, 0);
    for (int agent_id = 0; agent_id < population; agent_id++) {
      float x = bugPosition[2 * agent_id + 0] * patch + halfPatch;
      float y = bugPosition[2 * agent_id + 1] * patch + halfPatch;
      agent_vertex[12 * agent_id + 0 ] = x + fp[0].s[0];
      agent_vertex[12 * agent_id + 1 ] = y + fp[0].s[1];
      agent_vertex[12 * agent_id + 2 ] = 0;
      agent_vertex[12 * agent_id + 3 ] = x + fp[1].s[0];
      agent_vertex[12 * agent_id + 4 ] = y + fp[1].s[1];
      agent_vertex[12 * agent_id + 5 ] = 0;
      agent_vertex[12 * agent_id + 6 ] = x + fp[2].s[0];
      agent_vertex[12 * agent_id + 7 ] = y + fp[2].s[1];
      agent_vertex[12 * agent_id + 8 ] = 0;
      agent_vertex[12 * agent_id + 9 ] = x + fp[3].s[0];
      agent_vertex[12 * agent_id + 10] = y + fp[3].s[1];
      agent_vertex[12 * agent_id + 11] = 0;
    }

    glGenBuffers(1, &agent_vertex_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, agent_vertex_vbo);
    glBufferData(GL_ARRAY_BUFFER, agent_vbo_size, &agent_vertex[0], GL_STATIC_DRAW);
    glFinish();
    agent_vertex.clear();

    size_t space_vbo_size = sizeof(float) * 3 * 4 * width * height;
    std::vector<float> space_vertex(width * height * 3 * 4, 0);
    for (int i = 0; i < width * height; i++) {
      int x = i % width;
      int y = i / width;
      space_vertex[12 * i + 0 ] = x * patch;
      space_vertex[12 * i + 1 ] = y * patch;
      space_vertex[12 * i + 2 ] = 0;
      space_vertex[12 * i + 3 ] = x * patch + patch;
      space_vertex[12 * i + 4 ] = y * patch;
      space_vertex[12 * i + 5 ] = 0;
      space_vertex[12 * i + 6 ] = x * patch + patch;
      space_vertex[12 * i + 7 ] = y * patch + patch;
      space_vertex[12 * i + 8 ] = 0;
      space_vertex[12 * i + 9 ] = x * patch;
      space_vertex[12 * i + 10] = y * patch + patch;
      space_vertex[12 * i + 11] = 0;
    }

    glGenBuffers(1, &space_vertex_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, space_vertex_vbo);
    glBufferData(GL_ARRAY_BUFFER, space_vbo_size, &space_vertex[0], GL_STATIC_DRAW);
    glFinish();
    space_vertex.clear();
  }

  void createColorObjs() {
    float rgb[3];
    mcl::Color::trans(rgb, mcl::Color::Red, 1);//white
    size_t agent_vbo_size = sizeof(float) * 4 * 3 * population;
    std::vector<float> agent_color(population * 4 * 3, 0);
    for (int agent_id = 0; agent_id < population; agent_id++) {
      agent_color[12 * agent_id + 0 ] = rgb[0]; // r
      agent_color[12 * agent_id + 1 ] = rgb[1]; // g
      agent_color[12 * agent_id + 2 ] = rgb[2]; // b
      agent_color[12 * agent_id + 3 ] = rgb[0];
      agent_color[12 * agent_id + 4 ] = rgb[1];
      agent_color[12 * agent_id + 5 ] = rgb[2];
      agent_color[12 * agent_id + 6 ] = rgb[0];
      agent_color[12 * agent_id + 7 ] = rgb[1];
      agent_color[12 * agent_id + 8 ] = rgb[2];
      agent_color[12 * agent_id + 9 ] = rgb[0];
      agent_color[12 * agent_id + 10] = rgb[1];
      agent_color[12 * agent_id + 11] = rgb[2];
    }

    glGenBuffers(1, &agent_color_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, agent_color_vbo);
    glBufferData(GL_ARRAY_BUFFER, agent_vbo_size, &agent_color[0], GL_STATIC_DRAW);
    glFinish();
    agent_color.clear();

    mcl::Color::trans(rgb, mcl::Color::Red, -1);//black
    size_t space_vbo_size = sizeof(float) * 3 * 4 * width * height;
    std::vector<float> space_color(width * height * 4 * 3, 0);
    for (int i = 0; i < width * height; i++) {
      space_color[12 * i + 0 ] = rgb[0]; // r
      space_color[12 * i + 1 ] = rgb[1]; // g
      space_color[12 * i + 2 ] = rgb[2]; // b
      space_color[12 * i + 3 ] = rgb[0];
      space_color[12 * i + 4 ] = rgb[1];
      space_color[12 * i + 5 ] = rgb[2];
      space_color[12 * i + 6 ] = rgb[0];
      space_color[12 * i + 7 ] = rgb[1];
      space_color[12 * i + 8 ] = rgb[2];
      space_color[12 * i + 9 ] = rgb[0];
      space_color[12 * i + 10] = rgb[1];
      space_color[12 * i + 11] = rgb[2];
    }

    glGenBuffers(1, &space_color_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, space_color_vbo);
    glBufferData(GL_ARRAY_BUFFER, space_vbo_size, &space_color[0], GL_STATIC_DRAW);
    glFinish();
    space_color.clear();
  }

  int createCLBuffers() {
    cl_int ret;

    memSpace = clCreateBuffer(context,
			CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
			sizeof(int) * width * height,
			&space.front(),
			&ret);
		if (ret != CL_SUCCESS) return 0;
		buffers.push_back(memSpace);

    std::vector<float> zeroData(width * height, 0);

    memSpaceTemp1 = clCreateBuffer(context,
      CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
      sizeof(float) * width * height,
      &zeroData.front(),
      &ret);
    if (ret != CL_SUCCESS) return 0;
		buffers.push_back(memSpaceTemp1);

    memSpaceTemp2 = clCreateBuffer(context,
      CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
      sizeof(float) * width * height,
      &zeroData.front(),
      &ret);
    if (ret != CL_SUCCESS) return 0;
		buffers.push_back(memSpaceTemp2);

    memBugPosition = clCreateBuffer(context,
      CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
      sizeof(cl_int2) * population,
      &bugPosition.front(),
      &ret);
    if (ret != CL_SUCCESS) return 0;
		buffers.push_back(memBugPosition);

    memBugIdealTemp = clCreateBuffer(context,
      CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
      sizeof(int) * population,
      &bugIdealTemp.front(),
      &ret);
    if (ret != CL_SUCCESS) return 0;
		buffers.push_back(memBugIdealTemp);

    memBugOutputHeat = clCreateBuffer(context,
      CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
      sizeof(int) * population,
      &bugOutputHeat.front(),
      &ret);
    if (ret != CL_SUCCESS) return 0;
		buffers.push_back(memBugOutputHeat);

    std::vector<unsigned int> seedData(population * 4, 0);
    for (int i = 0; i < population * 4; i++) {
      seedData[i] = mcl::Random::random();
    }

    memSeed = clCreateBuffer(context,
      CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
      sizeof(cl_uint4) * population,
      &seedData.front(),
      &ret);
    if (ret != CL_SUCCESS) return 0;
		buffers.push_back(memSeed);

    memVertexObj = clCreateFromGLBuffer(context,
      CL_MEM_WRITE_ONLY,
      agent_vertex_vbo,
      &ret);
    if (ret != CL_SUCCESS) return 0;
		buffers.push_back(memVertexObj);

    memColorObj = clCreateFromGLBuffer(context,
      CL_MEM_WRITE_ONLY,
      space_color_vbo,
      &ret);
    if (ret != CL_SUCCESS) return 0;
		buffers.push_back(memColorObj);

    return 1;
  }

  int setKernelArgs() {
    cl_int ret = CL_SUCCESS;
    int i = 0;
    ret |= clSetKernelArg(writeAgentVertexObj, i++, sizeof(cl_mem), &memVertexObj);
    ret |= clSetKernelArg(writeAgentVertexObj, i++, sizeof(cl_mem), &memBugPosition);
    ret |= clSetKernelArg(writeAgentVertexObj, i++, sizeof(float), &patch);
    if (ret != CL_SUCCESS) return 0;

    ret = clSetKernelArg(writeSpaceColorObj, 0, sizeof(cl_mem), &memColorObj);
    if (ret != CL_SUCCESS) return 0;

    i = 0;
    ret |= clSetKernelArg(moveToIdealTempSpot, i++, sizeof(cl_mem), &memSeed);
    ret |= clSetKernelArg(moveToIdealTempSpot, i++, sizeof(cl_mem), &memSpace);
    ret |= clSetKernelArg(moveToIdealTempSpot, i++, sizeof(cl_mem), &memBugPosition);
    ret |= clSetKernelArg(moveToIdealTempSpot, i++, sizeof(cl_mem), &memBugIdealTemp);
    ret |= clSetKernelArg(moveToIdealTempSpot, i++, sizeof(cl_mem), &memBugOutputHeat);
    i++;
    ret |= clSetKernelArg(moveToIdealTempSpot, i++, sizeof(float), &randomMoveProb);
    if (ret != CL_SUCCESS) return 0;

    return 1;
  }

  int getOneDimIdx(int x, int y) {
    return x + width * y;
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
        width, height, population, ifps, ticks, scale_size);

      glutSetWindowTitle(tmp);
      fpsCount = 0;
      fpsLimit = (ifps > 1.f) ? (int)ifps : 1;

      start_clock = clock();
    }
  }

  /* GLUT Callback */
public:
  static void _wrapDisplay() { Heatbugs* ptr = *(get_ptr()); ptr->display(); }
  static void _wrapIdle() { Heatbugs* ptr = *(get_ptr()); ptr->idle(); }
  static void _wrapKeyboard(unsigned char key, int x, int y) { Heatbugs* ptr = *(get_ptr()); ptr->keyboard(key, x, y); }
  static void _wrapMouse(int button, int state, int x, int y) { Heatbugs* ptr = *(get_ptr()); ptr->mouse(button, state, x, y); }
  static void _wrapMotion(int x, int y) { Heatbugs* ptr = *(get_ptr()); ptr->motion(x, y); }
private:
  static Heatbugs** get_ptr() {
    static Heatbugs* ptr;
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

    glBindBuffer(GL_ARRAY_BUFFER, space_vertex_vbo);
    glVertexPointer(3, GL_FLOAT, 0, 0);

    glBindBuffer(GL_ARRAY_BUFFER, space_color_vbo);
    glColorPointer(3, GL_FLOAT, 0, 0);

    glDrawArrays(GL_QUADS, 0, width * height * 4);

    glBindBuffer(GL_ARRAY_BUFFER, agent_vertex_vbo);
    glVertexPointer(3, GL_FLOAT, 0, 0);

    glBindBuffer(GL_ARRAY_BUFFER, agent_color_vbo);
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

};

int main() {
  const int window_width  = 1000;
  const int window_height = 1000;

  const int width  = 1000 * 2;
  const int height = 1000 * 2;
  const int num_agents = 40000;

  Heatbugs(window_width,
    window_height,
    width,
    height,
    num_agents);

  std::cout << "Simulation finished\n";

  return 0;
}

