//#include "ColorHelper.h"
#include "SakiyaMas.h"

#include <iostream>
#include <vector>
#include <ctime>
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

class Segregation {
  const int window_width;
  const int window_height;
  const int width;
  const int height;
  const int num_teams;
  int population;
  int one_team_population;
  const float rate_friend;

  GLuint agentVertexObj;
  GLuint agentColorObj;

  const int max_num_teams; // 5
  const int empty;

  float patch;
  float halfPatch;

  std::vector<mcl::Color::ColorElement> color_set;

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
  cl_kernel moveToEmptySpot;
  cl_kernel writeVertexObj;
  cl_mem memSpace;
  cl_mem memPosition;
  cl_mem memVertexObj;
  cl_mem memSeed;
  std::vector<cl_mem> buffers;

public:
  Segregation(int window_width_,
    int window_height_,
    int width_,
    int height_,
    int num_teams_,
    double rate_population,
    double rate_friend_)
    : window_width(window_width_),
    window_height(window_height_),
    width(width_),
    height(height_),
    num_teams(num_teams_),
    rate_friend((float)rate_friend_),
    agentVertexObj(0),
    agentColorObj(0),
    max_num_teams(5),
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
    moveToEmptySpot(0),
    writeVertexObj(0)
  {
    patch = (float)window_width / width;
    halfPatch = patch / 2;

    /* Calculating population */
    one_team_population = width * height * rate_population / num_teams;
    population = one_team_population * num_teams;
  }

  ~Segregation() {
    cleanup();
  }

  int init() {
    if (num_teams > max_num_teams) {
      std::cerr << "Error: Invalid Number of Teams\n";
      return 0;
    }

    if (!initGlut()) {
      std::cerr << "Error: Initializing GLUT\n";
      return 0;
    }

    if (!initOpenCL()) {
      std::cerr << "Error: Initializing OpenCL\n";
      return 0;
    }

    /* Initializing Parameters */
    std::vector<int> positions(population * 2, 0);
    std::vector<int> groups(population, 0);

    std::vector<int> space(width * height, empty);

    int agent_id = 0;
    while (agent_id < population) {
      int x = mcl::Random::random(width);
      int y = mcl::Random::random(height);
      if (space[getOneDimIdx(x, y)] == empty) {
        positions[2 * agent_id + 0] = x, positions[2 * agent_id + 1] = y;
        space[getOneDimIdx(x, y)] = groups[agent_id] = agent_id % num_teams;
        agent_id++;
      }
    }

    /* Setting Color */
    color_set = std::vector<mcl::Color::ColorElement>(max_num_teams);
    color_set[0].value = mcl::Color::Red;
    color_set[1].value = mcl::Color::Green;
    color_set[2].value = mcl::Color::Yellow;
    color_set[3].value = mcl::Color::Blue;
    color_set[4].value = mcl::Color::Orange;

    createAgentBufferObjs(&positions.front(), &groups.front());
   
    if (!createCLBuffers(&space.front(), &positions.front(), &groups.front())) {
      return 0;
    }

    if (!setKernelArgs()) {
      return 0;
    }

    return 1;
  }

private:
  int run() {
    size_t gwSize = population;

    int err;
    err = launchKernel(&moveToEmptySpot, &gwSize, 1);
    if (err == 0) return 0;

    err = acquireGLObject(memVertexObj);
    if (err == 0) return 0;

    err = launchKernel(&writeVertexObj, &gwSize, 1);
    if (err == 0) return 0;

    err = releaseGLObject(memVertexObj);
    if (err == 0) return 0;

    return 1;
  }

  void createAgentBufferObjs(int *pos, int *group) {
    cl_float2 fp[4] = {0};
    fp[0].s[0] = halfPatch;      fp[0].s[1] = -1 * halfPatch;
    fp[1].s[0] = -1 * halfPatch; fp[1].s[1] = -1 * halfPatch;
    fp[2].s[0] = -1 * halfPatch; fp[2].s[1] = halfPatch;
    fp[3].s[0] = halfPatch;      fp[3].s[1] = halfPatch;

    size_t size = sizeof(float) * 4 * 3 * population;
    std::vector<float> bufferObj(population * 4 * 3);
    for (int i = 0; i < population; i++) {
      float x = pos[2 * i + 0] * patch + halfPatch;
      float y = pos[2 * i + 1] * patch + halfPatch;
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

    for (int i = 0; i < population; i++) {
      float rgb[3];
      mcl::Color::trans(rgb, color_set[ group[i] ].value, 0);
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

  int createCLBuffers(int *space, int *pos, int *group) {
    memSpace = createCLBuffer(space, width * height);

    memPosition = createCLBuffer(pos, population * 2);
    
    memVertexObj = createFromGLBuffer(agentVertexObj);
    
    std::vector<unsigned int> tmpSeed(population * 4, 0);
    for (int i = 0; i < population * 4; i++) {
      tmpSeed[i] = mcl::Random::random();
    }
    
    memSeed = createCLBuffer(&tmpSeed.front(), tmpSeed.size());
    
    for (size_t i = 0; i < buffers.size(); i++) {
      if (buffers[i] == 0) return 0;
    }

    return 1;
  }

  int setKernelArgs() {
    cl_int ret = CL_SUCCESS;
    int i = 0;
    ret |= clSetKernelArg(moveToEmptySpot, i++, sizeof(cl_mem), &memSpace);
    ret |= clSetKernelArg(moveToEmptySpot, i++, sizeof(cl_mem), &memSeed);
    ret |= clSetKernelArg(moveToEmptySpot, i++, sizeof(cl_mem), &memPosition);
    ret |= clSetKernelArg(moveToEmptySpot, i++, sizeof(float), &rate_friend);
    if (ret != CL_SUCCESS) {
      std::cerr << "Error: clSetKernelArg\n";
      return 0;
    }
    i = 0;
    ret |= clSetKernelArg(writeVertexObj, i++, sizeof(cl_mem), &memVertexObj);
    ret |= clSetKernelArg(writeVertexObj, i++, sizeof(cl_mem), &memPosition);
    ret |= clSetKernelArg(writeVertexObj, i++, sizeof(float), &patch);
    if (ret != CL_SUCCESS) {
      std::cerr << "Error: clSetKernelArg\n";
      return 0;
    }

    return 1;
  }

  int getOneDimIdx(int x, int y) {
    return x + width * y;
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

    moveToEmptySpot = clCreateKernel(program, "moveToEmptySpot", &ret);
    if (ret != CL_SUCCESS) {
      std::cerr << "Error " << ret << ": clCreateKernel\n";
      return 0;
    }

    writeVertexObj = clCreateKernel(program, "writeVertexObj", &ret);
    if (ret != CL_SUCCESS) {
      std::cerr << "Error " << ret << ": clCreateKernel\n";
      return 0;
    }

    return 1;
  }

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

  void start() {
#if defined(_WIN32)||(_WIN64)
    if (glewGetExtension("WGL_EXT_swap_control")) {
      wglSwapIntervalEXT(vsync);
    }
#endif
    Segregation** seg_ptr = get_ptr();
    *seg_ptr = this;
    glutDisplayFunc(_wrapDisplay);
    glutKeyboardFunc(_wrapKeyboard);
    glutIdleFunc(_wrapIdle);
    glutMouseFunc(_wrapMouse);
    glutMotionFunc(_wrapMotion);

    start_clock = std::clock();
    glutMainLoop();
  }

  /* GLUT Callback */
  static void _wrapDisplay() { Segregation* ptr = *(get_ptr()); ptr->display(); }
  static void _wrapIdle() { Segregation* ptr = *(get_ptr()); ptr->idle(); }
  static void _wrapKeyboard(unsigned char key, int x, int y) { Segregation* ptr = *(get_ptr()); ptr->keyboard(key, x, y); }
  static void _wrapMouse(int button, int state, int x, int y) { Segregation* ptr = *(get_ptr()); ptr->mouse(button, state, x, y); }
  static void _wrapMotion(int x, int y) { Segregation* ptr = *(get_ptr()); ptr->motion(x, y); }

private:
  static Segregation** get_ptr() {
    static Segregation* ptr;
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
    if (!buffers.empty()) {
      for (size_t i = 0; i < buffers.size(); i++) {
        if (buffers[i] != 0) clReleaseMemObject(buffers[i]);
      }
    }
    if (moveToEmptySpot != 0) clReleaseKernel(moveToEmptySpot);
    if (writeVertexObj != 0) clReleaseKernel(writeVertexObj);
    if (program != 0) clReleaseProgram(program);
    if (queue != 0) clReleaseCommandQueue(queue);
    if (context != 0) clReleaseContext(context);
  }
};

int main(int argc, char* argv[]) {
  const int window_width  = 1000;
  const int window_height = 1000;

  const int width  = 1000;
  const int height = 1000;
  const int num_teams = 5;
  const double rate_population = 0.75;
  const double rate_friend = 0.55;

  Segregation segregation(
    window_width,
    window_height,
    width,
    height,
    num_teams,
    rate_population,
    rate_friend
    );

  if (!segregation.init()) {
    return 1;
  }

  //segregation.setVsync(0);
  //segregation.setStop(true);
  segregation.start();

  std::cout << "Simulation finished\n";

  return 0;
}
