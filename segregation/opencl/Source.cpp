//#include "ColorHelper.h"
#include "../../common/SakiyaMas.h"

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

template <class T>
struct Point {
	T x;
	T y;
	Point() : x(0), y(0) {}
	Point(T x_, T y_) : x(x_), y(y_) {}
};

class Segregation {
	const int window_width;
	const int window_height;
	const int width;
	const int height;
	const int num_teams;
	int population;
	int one_team_population;
	const float rate_friend;

	GLuint agent_vertex_vbo;
	GLuint agent_color_vbo;

	const int max_num_teams; // 5
	const int empty;

	float patch;
	float halfPatch;

	std::vector< Point<int> > pos;
	std::vector<int> group;
	std::vector<int> space;

	Point<float> fp[4];
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
	cl_mem memGroup;
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
		agent_vertex_vbo(0),
		agent_color_vbo(0),
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
		writeVertexObj(0),
		memSpace(0),
		memPosition(0),
		memGroup(0),
		memVertexObj(0),
		memSeed(0)
	{
		if (num_teams > max_num_teams) {
			std::cerr << "Error: Invalid Number of Teams\n";
			std::exit(0);
		}

		if (!initGlut()) {
			std::exit(0);
		}

		if (!initOpenCL()) {
			std::exit(0);
		}

		patch = (float)window_width / width;
		halfPatch = patch / 2;

		/* Calculating population */
		one_team_population = width * height * rate_population / num_teams;
		population = one_team_population * num_teams;

		/* Initializing Parameters */
		pos = std::vector< Point<int> >(population, Point<int>());
		group = std::vector<int>(population, 0);
		space = std::vector<int>(width * height, empty);// empty: -1

		int agent_id = 0;
		while (agent_id < population) {
			int x = mcl::Random::random(width);
			int y = mcl::Random::random(height);
			if (space[getOneDimIdx(x, y)] == empty) {
				pos[agent_id].x = x, pos[agent_id].y = y;
				space[getOneDimIdx(x, y)] = group[agent_id] = agent_id % num_teams;
				agent_id++;
			}
		}

		/* Setting Positions on Screen */
		fp[0].x = halfPatch;      fp[0].y = -1 * halfPatch;
		fp[1].x = -1 * halfPatch; fp[1].y = -1 * halfPatch;
		fp[2].x = -1 * halfPatch; fp[2].y = halfPatch;
		fp[3].x = halfPatch;      fp[3].y = halfPatch;

		/* Setting Color */
		color_set = std::vector<mcl::Color::ColorElement>(max_num_teams);
		color_set[0].value = mcl::Color::Red;
		color_set[1].value = mcl::Color::Green;
		color_set[2].value = mcl::Color::Yellow;
		color_set[3].value = mcl::Color::Blue;
		color_set[4].value = mcl::Color::Orange;

		createVertexBuffer();
		createColorBuffer();

		createCLBuffers();

        pos.clear();
        group.clear();
        space.clear();

		setKernelArgs();
	}

	~Segregation() {
		if (agent_vertex_vbo != 0) {
			glBindBuffer(1, agent_vertex_vbo);
			glDeleteBuffers(1, &agent_vertex_vbo);
		}
		if (agent_color_vbo != 0) {
			glBindBuffer(1, agent_color_vbo);
			glDeleteBuffers(1, &agent_color_vbo);
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

	void setVsync(bool arg) {
		if (arg) vsync = 1;
		else vsync = 0;
	}

	void setStop(bool arg) {
		if (arg) pause = true;
	}

private:
	int run() {
		size_t gwSize = population;
		cl_event event_;

		cl_int ret;
		ret = clEnqueueNDRangeKernel(queue,
			moveToEmptySpot,
			1,
			NULL,
			&gwSize,
			NULL,
			0,
			NULL,
			&event_);
		if (ret != CL_SUCCESS) return 0;

		ret = clWaitForEvents(1, &event_);
		if (ret != CL_SUCCESS) return 0;
		clReleaseEvent(event_);

		ret = clEnqueueAcquireGLObjects(queue,
			1,
			&memVertexObj,
			0,
			NULL,
			&event_);
		if (ret != CL_SUCCESS) return 0;

		ret = clWaitForEvents(1, &event_);
		if (ret != CL_SUCCESS) return 0;
		clReleaseEvent(event_);

		ret = clEnqueueNDRangeKernel(queue,
			writeVertexObj,
			1,
			NULL,
			&gwSize,
			NULL,
			0,
			NULL,
			&event_);
		if (ret != CL_SUCCESS) return 0;

		ret = clWaitForEvents(1, &event_);
		if (ret != CL_SUCCESS) return 0;
		clReleaseEvent(event_);

		ret = clEnqueueReleaseGLObjects(queue,
			1,
			&memVertexObj,
			0,
			NULL,
			&event_);
		if (ret != CL_SUCCESS) return 0;

		ret = clWaitForEvents(1, &event_);
		if (ret != CL_SUCCESS) return 0;
		clReleaseEvent(event_);

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
		//ss << "#define PATCH (" << patch << ")\n";

		std::string newSrcStdStr = ss.str() + srcStdStr;
		const char *srcStr = newSrcStdStr.c_str();
		program = clCreateProgramWithSource(context, 1, (const char**)&srcStr, NULL, &ret);
		if (ret != CL_SUCCESS) return 0;

		ret = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
		if (ret != CL_SUCCESS) return 0;

		moveToEmptySpot = clCreateKernel(program, "moveToEmptySpot", &ret);
		if (ret != CL_SUCCESS) return 0;
		writeVertexObj = clCreateKernel(program, "writeVertexObj", &ret);
		if (ret != CL_SUCCESS) return 0;

		return 1;
	}

	void createVertexBuffer() {
		size_t agent_vbo_size = sizeof(float) * 4 * 3 * population;
		std::vector<float> agent_vertex(population * 4 * 3, 0);
		for (int agent_id = 0; agent_id < population; agent_id++) {
			agent_vertex[12 * agent_id + 0 ] = pos[agent_id].x * patch + halfPatch + fp[0].x;
			agent_vertex[12 * agent_id + 1 ] = pos[agent_id].y * patch + halfPatch + fp[0].y;
			agent_vertex[12 * agent_id + 2 ] = 0;
			agent_vertex[12 * agent_id + 3 ] = pos[agent_id].x * patch + halfPatch + fp[1].x;
			agent_vertex[12 * agent_id + 4 ] = pos[agent_id].y * patch + halfPatch + fp[1].y;
			agent_vertex[12 * agent_id + 5 ] = 0;
			agent_vertex[12 * agent_id + 6 ] = pos[agent_id].x * patch + halfPatch + fp[2].x;
			agent_vertex[12 * agent_id + 7 ] = pos[agent_id].y * patch + halfPatch + fp[2].y;
			agent_vertex[12 * agent_id + 8 ] = 0;
			agent_vertex[12 * agent_id + 9 ] = pos[agent_id].x * patch + halfPatch + fp[3].x;
			agent_vertex[12 * agent_id + 10] = pos[agent_id].y * patch + halfPatch + fp[3].y;
			agent_vertex[12 * agent_id + 11] = 0;
		}

		glGenBuffers(1, &agent_vertex_vbo);
		glBindBuffer(GL_ARRAY_BUFFER, agent_vertex_vbo);
		glBufferData(GL_ARRAY_BUFFER, agent_vbo_size, &agent_vertex[0], GL_STATIC_DRAW);
		glFinish();
	}

	void createColorBuffer() {
		size_t agent_vbo_size = sizeof(float) * 4 * 3 * population;
		std::vector<float> agent_color(population * 4 * 3, 0);
		for (int agent_id = 0; agent_id < population; agent_id++) {
			float rgb[3];
			mcl::Color::trans(rgb, color_set[ group[agent_id] ].value, 0);
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

		std::vector<int> tmpPos(population * 2, 0);
		for (int i = 0; i < population; i++) {
			tmpPos[2 * i + 0] = pos[i].x;
			tmpPos[2 * i + 1] = pos[i].y;
		}
		memPosition = clCreateBuffer(context,
			CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
			sizeof(cl_int2) * population,
			&tmpPos.front(),
			&ret);
		if (ret != CL_SUCCESS) return 0;
		buffers.push_back(memPosition);

		memGroup = clCreateBuffer(context,
			CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
			sizeof(int) * population,
			&group.front(),
			&ret);
		if (ret != CL_SUCCESS) return 0;
		buffers.push_back(memGroup);

		memVertexObj = clCreateFromGLBuffer(context, CL_MEM_WRITE_ONLY, agent_vertex_vbo, &ret);
		if (ret != CL_SUCCESS) return 0;
		buffers.push_back(memVertexObj);

		std::vector<unsigned int> tmpSeed(population * 4, 0);
		for (int i = 0; i < population * 4; i++) {
			tmpSeed[i] = mcl::Random::random();
		}
		memSeed = clCreateBuffer(context,
			CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
			sizeof(cl_uint4) * population,
			&tmpSeed.front(),
			&ret);
		if (ret != CL_SUCCESS) return 0;
		buffers.push_back(memSeed);

		return 1;
	}

	int setKernelArgs() {
		cl_int ret = CL_SUCCESS;
		int i = 0;
		ret |= clSetKernelArg(moveToEmptySpot, i++, sizeof(cl_mem), &memSpace);
		ret |= clSetKernelArg(moveToEmptySpot, i++, sizeof(cl_mem), &memSeed);
		ret |= clSetKernelArg(moveToEmptySpot, i++, sizeof(cl_mem), &memPosition);
		ret |= clSetKernelArg(moveToEmptySpot, i++, sizeof(cl_mem), &memGroup);
		ret |= clSetKernelArg(moveToEmptySpot, i++, sizeof(float), &rate_friend);
		if (ret != CL_SUCCESS) return 0;

		i = 0;
		ret |= clSetKernelArg(writeVertexObj, i++, sizeof(cl_mem), &memVertexObj);
		ret |= clSetKernelArg(writeVertexObj, i++, sizeof(cl_mem), &memPosition);
		ret |= clSetKernelArg(writeVertexObj, i++, sizeof(float), &patch);
		if (ret != CL_SUCCESS) return 0;

		return 1;
	}

	int getOneDimIdx(int x, int y) {
		return x + width * y;
	}

	int getOneDimIdx(Point<int> &point) {
		return point.x + width * point.y;
	}

	Point<int> getTorus(Point<int> &point) {
		Point<int> ret;
		ret.x = (point.x + width) % width;
		ret.y = (point.y + height) % height;
		return ret;
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

	const int width  = 1000 * 2;
	const int height = 1000 * 2;
	const int num_teams = 2;
	const double rate_population = 0.75;
	const double rate_friend = 0.75;

	Segregation segregation(
		window_width,
		window_height,
		width,
		height,
		num_teams,
		rate_population,
		rate_friend
		);
	segregation.setVsync(1);
	segregation.setStop(true);
	segregation.start();

	std::cout << "Simulation finished\n";

	return 0;
}
