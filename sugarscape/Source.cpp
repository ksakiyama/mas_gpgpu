//#include "ColorHelper.h"
#include "SakiyaMas.h"

#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <iterator>

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

  GLuint agent_vertex_vbo;
  GLuint agent_color_vbo;
  GLuint space_vertex_vbo;
  GLuint space_color_vbo;

  const int empty;

  float patch;
  float halfPatch;

  //std::vector< Point<int> > pos;
  std::vector<int> pos;
  std::vector<int> age;
  std::vector<int> max_age;
  std::vector<int> vision;
  std::vector<int> sugar;
  std::vector<int> metabolism;

  std::vector<int> space;
  std::vector<int> space_max_sugar;

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

  cl_kernel kernelMoveToBestSugarSpot;
  cl_kernel kernelUpdateAgentParameter;
  cl_kernel kernelGrowupSpaceSugar;
  cl_kernel kernelWriteAgentVertexObj;
  cl_kernel kernelWriteSpaceColorObj;

  std::vector<cl_kernel*> kernels;

  cl_mem memSpace;
  cl_mem memSeed;
  cl_mem memPosition;
  cl_mem memAge;
  cl_mem memMaxAge;
  cl_mem memVision;
  cl_mem memSugar;
  cl_mem memMetabolism;
  cl_mem memSpaceSugar;
  cl_mem memSpaceMaxSugar;
  cl_mem memAgentVertexObj;
  cl_mem memSpaceColorObj;

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
    kernelMoveToBestSugarSpot(0),
    kernelUpdateAgentParameter(0),
    kernelWriteAgentVertexObj(0),
    kernelGrowupSpaceSugar(0),
    kernelWriteSpaceColorObj(0),
    memSpace(0),
    memSeed(0),
    memPosition(0),
    memAge(0),
    memMaxAge(0),
    memVision(0),
    memSugar(0),
    memMetabolism(0),
    memSpaceSugar(0),
    memSpaceMaxSugar(0),
    memAgentVertexObj(0),
    memSpaceColorObj(0)
  {
    if (!initGlut()) {
      std::cerr << "Error: Initializing GLUT\n";
      return;
    }

    if (!initOpenCL()) {
      std::cerr << "Error: Initializing OpenCL\n";
      return;
    }

    patch = (float)window_width / width;
    halfPatch = patch / 2;

    /* Initializing Parameters */
    space = std::vector<int>(width * height, empty);
    space_max_sugar = std::vector<int>(width * height, 0);

    pos = std::vector<int>(population * 2, 0);
    age        = std::vector<int>(population, 0);
    max_age    = std::vector<int>(population, 0);
    vision     = std::vector<int>(population, 0);
    sugar      = std::vector<int>(population, 0);
    metabolism = std::vector<int>(population, 0);

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

    if (!readMapFile()) {

      std::cerr << "Error: Reading TXT File\n";
      return;
    }

    createVertexBuffers();
    createColorBuffers();

    if (!createCLBuffers()) {
      std::cerr << "Error: Creating OpenCL Buffers\n";
      return;
    }
  }

  ~Sugarscape() {
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
    if (!kernels.empty()) {
      for (size_t i = 0; i < kernels.size(); i++) {
        if (*kernels[i] != 0) clReleaseKernel(*kernels[i]);
        else std::cerr << "debug\n";
      }
    }
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

  void setVsync(bool arg) {
    if (arg) vsync = 1;
    else vsync = 0;
  }

  void setStop(bool arg) {
    if (arg) pause = true;
    else pause = false;
  }

private:
  int run() {
    //TODO
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

    std::string newSrcStdStr = ss.str() + srcStdStr;
    const char *srcStr = newSrcStdStr.c_str();
    program = clCreateProgramWithSource(context, 1, (const char**)&srcStr, NULL, &ret);
    if (ret != CL_SUCCESS) return 0;

    ret = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (ret != CL_SUCCESS) return 0;

    kernelMoveToBestSugarSpot = clCreateKernel(program, "moveToBestSugarSpot", &ret);
    if (ret != CL_SUCCESS) return 0;
    kernels.push_back(&kernelMoveToBestSugarSpot);

    kernelUpdateAgentParameter = clCreateKernel(program, "updateAgentParameter", &ret);
    if (ret != CL_SUCCESS) return 0;
    kernels.push_back(&kernelUpdateAgentParameter);

    kernelGrowupSpaceSugar = clCreateKernel(program, "growupSpaceSugar", &ret);
    if (ret != CL_SUCCESS) return 0;
    kernels.push_back(&kernelGrowupSpaceSugar);

    kernelWriteAgentVertexObj = clCreateKernel(program, "writeAgentVertexObj", &ret);
    if (ret != CL_SUCCESS) return 0;
    kernels.push_back(&kernelWriteAgentVertexObj);

    kernelWriteSpaceColorObj = clCreateKernel(program, "writeSpaceColorObj", &ret);
    if (ret != CL_SUCCESS) return 0;
    kernels.push_back(&kernelWriteSpaceColorObj);

    return 1;
  }

  void createVertexBuffers() {
    cl_float2 fp[4];
    fp[0].s[0] = halfPatch;      fp[0].s[1] = 0;
    fp[1].s[0] = 0;              fp[1].s[1] = -1 * halfPatch;
    fp[2].s[0] = -1 * halfPatch; fp[2].s[1] = 0;
    fp[3].s[0] = 0;              fp[3].s[1] = halfPatch;

    size_t agent_vbo_size = sizeof(float) * 4 * 3 * population;
    std::vector<float> agent_vertex(population * 4 * 3, 0);
    for (int agent_id = 0; agent_id < population; agent_id++) {
      float x = pos[2 * agent_id + 0] * patch + halfPatch;
      float y = pos[2 * agent_id + 1] * patch + halfPatch;
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

    fp[0].s[0] = halfPatch;      fp[0].s[1] = -1 * halfPatch;
    fp[1].s[0] = -1 * halfPatch; fp[1].s[1] = -1 * halfPatch;
    fp[2].s[0] = -1 * halfPatch; fp[2].s[1] = halfPatch;
    fp[3].s[0] = halfPatch;      fp[3].s[1] = halfPatch;

    size_t space_vbo_size = sizeof(float) * 4 * 3 * width * height;
    std::vector<float> space_vertex(width * height * 4 * 3, 0);
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
  }

  void createColorBuffers() {
    float rgb[3];
    size_t agent_vbo_size = sizeof(float) * 4 * 3 * population;
    std::vector<float> agent_color(population * 4 * 3, 0);
    mcl::Color::trans(rgb, mcl::Color::Red, 0);
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

    size_t space_vbo_size = sizeof(float) * 4 * 3 * width * height;
    std::vector<float> space_color(width * height * 4 * 3, 0);
    mcl::Color::trans(rgb, mcl::Color::Yellow, 0);
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
  }

  int createCLBuffers() {
    return 1;
  }

  int readMapFile() {
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
        space_max_sugar[ getOneDimIdx(x, y) ] = sugar[ getOneDimIdx(x, y) ];
      }
    }

    return 1;
  }

  template <class T>
  int setArg(cl_kernel *krnl, cl_uint num, T value) {
    cl_int ret = clSetKernelArg(
      *krnl,
      num,
      details::SetArgHandler<T>::size(value),//sizeof(T),
      details::SetArgHandler<T>::ptr(value));

    if (ret != CL_SUCCESS) return 0;

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
        window_width, window_height, population, ifps, ticks, scale_size);

      glutSetWindowTitle(tmp);
      fpsCount = 0;
      fpsLimit = (ifps > 1.f) ? (int)ifps : 1;

      start_clock = clock();
    }
  }

  /* GLUT Callback */
public:
  static void _wrapDisplay() { Sugarscape* ptr = *(get_ptr()); ptr->display(); }
  static void _wrapIdle() { Sugarscape* ptr = *(get_ptr()); ptr->idle(); }
  static void _wrapKeyboard(unsigned char key, int x, int y) { Sugarscape* ptr = *(get_ptr()); ptr->keyboard(key, x, y); }
  static void _wrapMouse(int button, int state, int x, int y) { Sugarscape* ptr = *(get_ptr()); ptr->mouse(button, state, x, y); }
  static void _wrapMotion(int x, int y) { Sugarscape* ptr = *(get_ptr()); ptr->motion(x, y); }
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

int main(int argc, char* argv[]) {
  const int window_width  = 1000;
  const int window_height = 1000;

  const int width  = 50;
  const int height = 50;
  const int num_agents = 50;

  Sugarscape sugarscape(
    window_width,
    window_height,
    width,
    height,
    num_agents
    );
  //sugarscape.setVsync(0);
  //sugarscape.setStop(true);
  sugarscape.start();

  std::cout << "Simulation finished\n";

  return 0;
}
