//#include "ColorHelper.h"
#include "../../common/SakiyaMas.h"

#include <iostream>
#include <vector>
#include <ctime>
#include <cstdio>
#include <cstdlib>
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
	const double rate_friend;

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
		rate_friend(rate_friend_),
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
		ticks(0)
	{
		if (num_teams > max_num_teams) {
			std::cerr << "Error: Invalid Number of Teams\n";
			std::exit(0);
		}

		if (!initGlut()) {
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
		space = std::vector<int>(width * height, -1);// empty: -1

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
	void run() {
		float *vertexObj;
		glBindBuffer(GL_ARRAY_BUFFER, agent_vertex_vbo);
		vertexObj = (float*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);

		for (int agent_id = 0; agent_id < population; agent_id++) {
			/* Search around */
			int count = 0;
			int friendCount = 0;
			for (int dy = -1; dy <= 1; dy++) {
			for (int dx = -1; dx <= 1; dx++) {
				if (dx == 0 && dy == 0) continue;
				Point<int> searchPos(pos[agent_id].x + dx, pos[agent_id].y + dy);
				searchPos = getTorus(searchPos);
				int searchIdx = getOneDimIdx(searchPos);
				if (space[searchIdx] != empty) {
					count++;
					if (space[searchIdx] == group[agent_id]) {
						friendCount++;
					}
				}
			}
			}

			/* Computing Friend Rate */
			if (count != 0) {
				double moveProb = (double)friendCount / count;
				/* Move */
				if (moveProb < rate_friend) {
					while(1) {
						int x = mcl::Random::random(width);
						int y = mcl::Random::random(height);
						if (space[ getOneDimIdx(x, y) ] == empty) {
							space[ getOneDimIdx(x, y) ] = group[agent_id];
							space[ getOneDimIdx(pos[agent_id].x, pos[agent_id].y) ] = empty;
							pos[agent_id] = Point<int>(x, y);
							break;
						}
					}

					/* Writing Vertex Buffer */
					float x = pos[agent_id].x * patch + halfPatch;
					float y = pos[agent_id].y * patch + halfPatch;
					vertexObj[12 * agent_id + 0 ] = x + fp[0].x;
					vertexObj[12 * agent_id + 1 ] = y + fp[0].y;
					//vertexObj[12 * agent_id + 2 ] = 0;
					vertexObj[12 * agent_id + 3 ] = x + fp[1].x;
					vertexObj[12 * agent_id + 4 ] = y + fp[1].y;
					//vertexObj[12 * agent_id + 5 ] = 0;
					vertexObj[12 * agent_id + 6 ] = x + fp[2].x;
					vertexObj[12 * agent_id + 7 ] = y + fp[2].y;
					//vertexObj[12 * agent_id + 8 ] = 0;
					vertexObj[12 * agent_id + 9 ] = x + fp[3].x;
					vertexObj[12 * agent_id + 10] = y + fp[3].y;
					//vertexObj[12 * agent_id + 11] = 0;
				}
			}
		}

		glUnmapBuffer(GL_ARRAY_BUFFER);
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
			run();
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

	const int width  = 1000/2;
	const int height = 1000/2;
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
	segregation.setVsync(0);
	segregation.setStop(true);
	segregation.start();

	std::cout << "Simulation finished\n";

	return 0;
}
