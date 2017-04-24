import threading
from tkinter import *
import time
import random
import copy
import math
import traceback

from grid import *
from particle import Particle
from utils import *
from setting import *


# GUI class
class GUIWindow():
    def __init__(self, grid):
        self.width = grid.width
        self.height = grid.height
        self.update_cnt = 0

        self.grid = grid
        self.running = threading.Event()
        self.updated = threading.Event()
        self.updated.clear()
        self.lock = threading.Lock()
        # grid info
        self.occupied = grid.occupied
        self.markers = grid.markers

        self.particles = []
        self.robot = None

        self.mean_x = None
        self.mean_y = None
        self.mean_heading = None
        self.mean_confident = None

        #print("Occupied: ")
        #print(self.occupied)
        #print("Markers: ")
        #print(self.markers)


    """
    plot
    """
    def drawGrid(self):
        for y in range(1,self.grid.height):
            self.canvas.create_line(0, y * self.grid.scale, int(self.canvas.cget("width")) - 1, y * self.grid.scale)
        for x in range(1,self.grid.width):
            self.canvas.create_line(x * self.grid.scale, 0, x * self.grid.scale, int(self.canvas.cget("height")) - 1)

    def drawOccubpied(self):
        for block in self.occupied:
            self.colorCell(block, '#222222')

    def drawMarkers(self):
        for marker in self.markers:
            marker_x, marker_y, marker_h = parse_marker_info(marker[0], marker[1], marker[2]);

            arrow_head_x, arrow_head_y = rotate_point(0.8, 0, marker_h)
            self.colorLine((marker_x, marker_y), (marker_x + arrow_head_x, marker_y + arrow_head_y), \
                linewidth=2, color='#222222')
            c1x, c1y = rotate_point(0.2, -0.5, marker_h)
            c2x, c2y = rotate_point(0, 0.5, marker_h)
            self.colorRectangle((marker_x+c1x, marker_y+c1y), (marker_x+c2x, marker_y+c2y), '#00FFFF')

    def weight_to_color(self, weight):
        return "#%02x00%02x" % (int(weight * 255), int((1 - weight) * 255))

    def _show_mean(self, x, y, heading_deg, confident=False):
        if x == None:
            return
        if confident:
            color = "#00AA00"
        else:
            color = "#CCCCCC"
        location = (x,y)
        robot_tag = 'robot'
        self.canvas.delete(robot_tag)
        self.colorTriangle(location, heading_deg, color,tri_size=20,tags=robot_tag)


    def _show_particles(self, particles):
        plot_cnt = PARTICLE_MAX_SHOW if len(particles) > PARTICLE_MAX_SHOW else len(particles)
        if not plot_cnt:
            return
        draw_skip = len(particles)/plot_cnt
        line_length = 0.3

        idx = 0
        while idx < len(particles):
            p = particles[int(idx)]
            coord = (p.x,p.y)
            # print((p.x,p.y))
            self.colorCircle(coord, '#FF0000', 2)
            ldx, ldy = rotate_point(line_length, 0, p.h)
            self.colorLine(coord, (coord[0]+ldx, coord[1]+ldy))
            idx += draw_skip

    def _show_robot(self, robot):
        coord = (robot.x, robot.y)
        self.colorTriangle(coord, robot.h, '#FF0000', tri_size=15)
        # plot fov
        fov_lx, fov_ly = rotate_point(8, 0, robot.h + ROBOT_CAMERA_FOV_DEG / 2)
        fov_rx, fov_ry = rotate_point(8, 0, robot.h - ROBOT_CAMERA_FOV_DEG / 2)
        self.colorLine(coord, (coord[0]+fov_lx, coord[1]+fov_ly), color='#222222', linewidth=2, dashed=True)
        self.colorLine(coord, (coord[0]+fov_rx, coord[1]+fov_ry), color='#222222', linewidth=2, dashed=True)

    def clean_world(self):
        #for eachparticle in self.dots:
        #    self.canvas.delete(eachparticle)
        self.canvas.delete("all")
        self.drawGrid()
        self.drawOccubpied()
        self.drawMarkers()

    """
    plot utils
    """

    # Draw a colored square at the specified grid coordinates
    def colorCell(self, location, color):
        coords = (location[0]*self.grid.scale, (self.height-location[1]-1)*self.grid.scale)
        self.canvas.create_rectangle(coords[0], coords[1], coords[0] + self.grid.scale, coords[1] + self.grid.scale, fill=color)

    def colorRectangle(self, corner1, corner2, color):
        coords1 =  (corner1[0]*self.grid.scale, (self.height-corner1[1])*self.grid.scale)
        coords2 =  (corner2[0]*self.grid.scale, (self.height-corner2[1])*self.grid.scale)
        self.canvas.create_rectangle(coords1[0], coords1[1], coords2[0], coords2[1], fill=color)

    def colorCircle(self,location, color, dot_size = 5):
        x0, y0 = location[0]*self.grid.scale - dot_size, (self.height-location[1])*self.grid.scale - dot_size
        x1, y1 = location[0]*self.grid.scale + dot_size, (self.height-location[1])*self.grid.scale + dot_size
        # print(x0,y0,x1,y1)
        return self.canvas.create_oval(x0, y0, x1, y1, fill=color)

    def colorLine(self, coord1, coord2, color='black', linewidth=1, dashed=False, tags=''):
        if dashed:
            self.canvas.create_line(coord1[0] * self.grid.scale, (self.height-coord1[1])* self.grid.scale, \
                coord2[0] * self.grid.scale, (self.height-coord2[1]) * self.grid.scale,  \
                fill=color, width=linewidth, dash=(5,3), tags=tags)
        else:
            self.canvas.create_line(coord1[0] * self.grid.scale, (self.height-coord1[1])* self.grid.scale, \
                coord2[0] * self.grid.scale, (self.height-coord2[1]) * self.grid.scale,  \
                fill=color, width=linewidth, tags=tags)

    def colorTriangle(self, location, heading_deg, color, tri_size, tags=None):
        hx, hy = rotate_point(tri_size, 0, heading_deg)
        lx, ly = rotate_point(-tri_size, tri_size, heading_deg)
        rx, ry = rotate_point(-tri_size, -tri_size, heading_deg)
        # reverse Y here since input to row, not Y
        hrot = (hx + location[0]*self.grid.scale, -hy + (self.height-location[1])*self.grid.scale)
        lrot = (lx + location[0]*self.grid.scale, -ly + (self.height-location[1])*self.grid.scale)
        rrot = (rx + location[0]*self.grid.scale, -ry + (self.height-location[1])*self.grid.scale)
        return self.canvas.create_polygon(hrot[0], hrot[1], lrot[0], lrot[1], rrot[0], rrot[1], \
            fill=color, outline='#000000',width=1,tags=tags)

    def colorsquare(self, location, color, bg=False, tags=''):
        """Draw a colored square at a given location

            Arguments:
            location -- coordinates of square
            color -- desired color, hexadecimal string (e.g.: '#C0FFEE')
            bg -- draw square in background, default False
            tags -- tags to apply to square, list of strings or string
        """
        coords = (location[0]*self.grid.scale, (self.grid.height - 1 - location[1])*self.grid.scale)
        rect = self.canvas.create_rectangle(coords[0], coords[1], coords[0] + self.grid.scale, coords[1] + self.grid.scale, fill=color, tags=tags)
        if bg:
            self.canvas.tag_lower(rect)

    def drawstart(self):
        """Redraw start square
            Color is green by default
        """
        self.canvas.delete('start')
        if self.grid._start != None:
            self.colorsquare(self.grid._start, '#00DD00', tags='start')


    def drawgoals(self):
        """Redraw all goal cells
            Color is blue by default
        """
        self.canvas.delete('goal')
        for goal in self.grid._goals:
            self.colorsquare(goal, '#0000DD', tags='goal')


    def drawallvisited(self):
        """Redraw all visited cells
            Color is light gray by default
        """
        
        self.canvas.delete('visited')
        for loc in self.grid._visited:
            self.colorsquare(loc, '#CCCCCC', bg = True, tags='visited')


    def drawnewvisited(self):
        """Draw any new visited cells added since last call
            Does not delete previously drawn visited cells
            Color is light gray by default
        """
        
        for loc in self.grid._newvisited:
            self.colorsquare(loc, '#CCCCCC', bg = True, tags='visited')
        self.grid._newvisited = []


    def drawobstacles(self):
        """Redraw all obstacles
            Color is dark gray by default
        """
        
        self.canvas.delete('obstacle')
        for obstacle in self.grid._obstacles:
            self.colorsquare(obstacle, '#222222', bg = True, tags='obstacle')


    def drawpathedge(self, start, end):
        """Draw a path segment between two cells

            Arguments:
            start -- starting coordinate
            end -- end coordinate
        """
        
        startcoords = ((start[0] + 0.5) * self.scale, (self.grid.height - (start[1] + 0.5)) * self.scale)
        endcoords = ((end[0] + 0.5) * self.scale, (self.grid.height - (end[1] + 0.5)) * self.scale)
        self.canvas.create_line(startcoords[0], startcoords[1], endcoords[0], endcoords[1], fill = '#DD0000', width = 5, arrow = LAST, tag='path')


    def drawpath(self):
        """Draw the grid's path, if any
        """
        
        self.canvas.delete('path')
        if len(self.grid._path) > 1:
            current = self.grid._path[0]
            for point in self.grid._path[1:]:
                self.drawpathedge(current, point)
                current = point


    def setup(self):
        """Do initial drawing of grid, start, goals, and obstacles
        """
        
        self.grid.lock.acquire()
        
        self.drawgoals()
        self.drawstart()
        self.drawobstacles()
        
        self.grid.lock.release()

    def setupdate(self):
        self.updateflag = True

    def update(self):
        
        #self.grid.lock.acquire()
        self._show_mean(self.mean_x, self.mean_y, self.mean_heading, self.mean_confident)
        
        # if 'path' in self.grid.changes:
        #     self.drawpath()
        # if 'visited' in self.grid.changes:
        #     self.drawnewvisited()
        # if 'allvisited' in self.grid.changes:
        #     self.drawallvisited()
        # if 'goals' in self.grid.changes:
        #     self.drawgoals()
        # if 'start' in self.grid.changes:
        #     self.drawstart()
        # if 'obstacles' in self.grid.changes:
        #     self.drawobstacles()

        # self.grid.changes = []
        # self.grid.updated.clear()
        # self.grid.lock.release()
        self.updated.clear()

    # start GUI thread
    def start(self):
        master = Tk()
        master.wm_title("Particle Filter: Grey/Green - estimated, Red - ground truth")

        self.canvas = Canvas(master, width = self.grid.width * self.grid.scale, height = self.grid.height * self.grid.scale, bd = 0, bg = '#FFFFFF')
        self.canvas.pack()

        self.drawGrid()
        self.drawOccubpied()
        self.drawMarkers()

        # Draw grid and any initial items
        self.setup()

        # Start mainloop and indicate that it is running
        self.running.set()
        while True:
            self.updated.wait()
            if self.updated.is_set():
                self.update()
            try:
                master.update_idletasks()
                master.update()
            except TclError:
                traceback.print_exc()
                break

        # Indicate that main loop has finished
        self.running.clear()
