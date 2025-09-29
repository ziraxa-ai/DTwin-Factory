from __future__ import annotations
import math
import sys
import time
import threading
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np

# Pygame + OpenGL
import pygame
from pygame.locals import DOUBLEBUF, OPENGL
from OpenGL.GL import *
from OpenGL.GLU import *

# --------------- Config ---------------
WINDOW_SIZE = (1280, 720)
GRID_ROWS, GRID_COLS = 5, 10  # 5x10 = 50
STATION_SPACING_X = 3.0
STATION_SPACING_Z = 4.0
FLOOR_SIZE = (50.0, 50.0)
USE_LIVE = True  # set True to hook to real simulator (see LiveSimThread below)

# --------------- Data Structures ---------------
StationState = str  # "idle"|"busy"|"down"|"blocked"|"quality"
STATE_COLORS: Dict[StationState, Tuple[float, float, float]] = {
    "idle": (0.63, 0.67, 0.75),
    "busy": (0.20, 0.82, 0.58),
    "down": (0.93, 0.27, 0.27),
    "blocked": (0.96, 0.62, 0.13),
    "quality": (0.38, 0.65, 0.98),
}

@dataclass
class StationSnap:
    id: int
    name: str
    state: StationState
    buffer: int
    wip: int

@dataclass
class Snapshot:
    t: float
    throughput: int
    wipBusy: int
    backlog: int
    inventory: Dict[str, int]
    stations: List[StationSnap]
    agv: List[Tuple[float, float, float]]  # x, y, z

# --------------- Demo Data Generator ---------------
def make_demo_snapshot(t: float, n: int = 50) -> Snapshot:
    stations: List[StationSnap] = []
    for i in range(n):
        r = np.random.rand()
        if r < 0.05:
            st = "down"
        elif r < 0.35:
            st = "busy"
        elif r < 0.45:
            st = "blocked"
        elif r < 0.50:
            st = "quality"
        else:
            st = "idle"
        buf = max(0, min(8, int(4 + 3 * math.sin(0.3 * t + i))))
        stations.append(StationSnap(i, f"ST{i:02d}", st, buf, 1 if st == "busy" else 0))
    agv = []
    for k in range(3):
        x = 10.0 * math.sin(0.5 * t + k)
        z = 6.0 * math.cos(0.5 * t + k)
        agv.append((x, 0.3, z))
    return Snapshot(
        t=t,
        throughput=int(t // 2),
        wipBusy=sum(1 for s in stations if s.state == "busy"),
        backlog=max(0, 200 - int(t // 2)),
        inventory={"engine": 100 + int(10 * math.sin(t/5)), "tire": 240, "seat": 160, "door": 140},
        stations=stations,
        agv=agv,
    )

# --------------- Live Simulator Hook (optional) ---------------
class LiveSimThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.snapshot: Optional[Snapshot] = None
        self.running = True

    def run(self):
        try:
            # Lazy import to avoid dependency if not needed
            from virtual_factory import Factory, FactoryEnv
            import random
            f = Factory(horizon=8*3600)
            env = FactoryEnv(f)
            env.reset()
            while self.running:
                action = {}
                if np.random.rand() < 0.05:
                    action['expedite_station'] = np.random.randint(0, f.n_stations)
                obs, rew, done, info = env.step(action)
                stations: List[StationSnap] = []
                for st in f.line.stations:
                    state: StationState
                    if st.is_broken(f.ts):
                        state = "down"
                    elif st.current is not None and st.is_busy(f.ts):
                        state = "busy"
                    elif st.current is not None and not st.is_busy(f.ts):
                        state = "blocked"
                    else:
                        state = "idle"
                    stations.append(StationSnap(st.cfg.idx, st.cfg.name, state, len(st.buffer), 1 if st.current else 0))
                agv = [(0.0, 0.3, 0.0)]
                self.snapshot = Snapshot(
                    t=float(f.ts),
                    throughput=len(f.collected_finished),
                    wipBusy=sum(1 for s in f.line.stations if s.current),
                    backlog=sum(v['backorders'] for v in f.inventory.state.values()),
                    inventory={p: s['on_hand'] for p, s in f.inventory.state.items()},
                    stations=stations,
                    agv=agv,
                )
                if done:
                    env.reset()
                time.sleep(1/30)
        except Exception as e:
            print("[LiveSimThread] Falling back to demo:", e)
            while self.running:
                self.snapshot = make_demo_snapshot(time.time() % 1000.0)
                time.sleep(1/30)

# --------------- OpenGL Helpers ---------------
VERT_SRC = """
#version 120
attribute vec3 aPos;
attribute vec3 aNormal;
uniform mat4 uMVP;
uniform mat4 uModel;
uniform mat3 uNormalMat;
varying vec3 vNormal;
varying vec3 vWorldPos;
void main(){
    vNormal = normalize(uNormalMat * aNormal);
    vec4 wp = uModel * vec4(aPos,1.0);
    vWorldPos = wp.xyz;
    gl_Position = uMVP * vec4(aPos,1.0);
}
"""
FRAG_SRC = """
#version 120
varying vec3 vNormal;
varying vec3 vWorldPos;
uniform vec3 uColor;
uniform vec3 uViewPos;
uniform vec3 uLightDir;
uniform vec3 uLightColor;
void main(){
    vec3 N = normalize(vNormal);
    vec3 L = normalize(-uLightDir);
    float diff = max(dot(N, L), 0.0);
    vec3 V = normalize(uViewPos - vWorldPos);
    vec3 R = reflect(-L, N);
    float spec = pow(max(dot(R, V), 0.0), 32.0);
    vec3 col = (0.10*uColor) + (0.9*diff*uColor) + (0.25*spec*uLightColor);
    gl_FragColor = vec4(col, 1.0);
}
"""

def compile_shader(src, shader_type):
    s = glCreateShader(shader_type)
    glShaderSource(s, src)
    glCompileShader(s)
    if glGetShaderiv(s, GL_COMPILE_STATUS) != GL_TRUE:
        raise RuntimeError(glGetShaderInfoLog(s))
    return s

def create_program(vs_src, fs_src):
    vs = compile_shader(vs_src, GL_VERTEX_SHADER)
    fs = compile_shader(fs_src, GL_FRAGMENT_SHADER)
    prog = glCreateProgram()
    glAttachShader(prog, vs)
    glAttachShader(prog, fs)
    glLinkProgram(prog)
    if glGetProgramiv(prog, GL_LINK_STATUS) != GL_TRUE:
        raise RuntimeError(glGetProgramInfoLog(prog))
    return prog

# --------------- Geometry (Cube) ---------------
def make_cube():
    # positions & normals for a unit cube centered at origin
    P = []
    N = []
    def face(nx, ny, nz, ex, ey):
        # create two triangles per face
        n = (nx, ny, nz)
        # basis vectors on the face plane
        for tri in [(( -1, -1), ( 1, -1), ( 1,  1)), (( -1, -1), ( 1,  1), ( -1,  1))]:
            for (u, v) in tri:
                x = nx + ex*u
                y = ny + ey*v
                z = nz
                P.append((x, y, z))
                N.append(n)
    # +Z face
    face(0,0, 0.5, 0.5, 0.5)
    # -Z
    face(0,0,-0.5, 0.5, 0.5)
    # +X
    def face_axis(nx, ny, nz, ax):
        if ax=='x':
            for tri in [((-0.5,-0.5,0.5),( -0.5,0.5,0.5),( -0.5,0.5,-0.5)),((-0.5,-0.5,0.5),(-0.5,0.5,-0.5),(-0.5,-0.5,-0.5))]:
                for x,y,z in tri:
                    P.append((x,y,z)); N.append((-1,0,0))
        if ax=='X':
            for tri in [((0.5,-0.5,0.5),(0.5,0.5,0.5),(0.5,0.5,-0.5)),((0.5,-0.5,0.5),(0.5,0.5,-0.5),(0.5,-0.5,-0.5))]:
                for x,y,z in tri:
                    P.append((x,y,z)); N.append((1,0,0))
        if ax=='y':
            for tri in [((-0.5,0.5,0.5),(0.5,0.5,0.5),(0.5,0.5,-0.5)),((-0.5,0.5,0.5),(0.5,0.5,-0.5),(-0.5,0.5,-0.5))]:
                for x,y,z in tri:
                    P.append((x,y,z)); N.append((0,1,0))
        if ax=='Y':
            for tri in [((-0.5,-0.5,0.5),(0.5,-0.5,0.5),(0.5,-0.5,-0.5)),((-0.5,-0.5,0.5),(0.5,-0.5,-0.5),(-0.5,-0.5,-0.5))]:
                for x,y,z in tri:
                    P.append((x,y,z)); N.append((0,-1,0))
    face_axis(0,0,0,'x'); face_axis(0,0,0,'X'); face_axis(0,0,0,'y'); face_axis(0,0,0,'Y')
    return np.array(P, dtype=np.float32), np.array(N, dtype=np.float32)

# --------------- Camera ---------------
class Camera:
    def __init__(self):
        self.target = np.array([0.0, 0.8, 0.0], dtype=np.float32)
        self.radius = 26.0
        self.theta = 0.8  # around Y
        self.phi = 0.9    # from top
        self.up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    def pos(self):
        x = self.target[0] + self.radius * math.cos(self.phi) * math.sin(self.theta)
        y = self.target[1] + self.radius * math.sin(self.phi)
        z = self.target[2] + self.radius * math.cos(self.phi) * math.cos(self.theta)
        return np.array([x, y, z], dtype=np.float32)

# --------------- Math helpers ---------------
def perspective(fovy, aspect, znear, zfar):
    f = 1.0 / math.tan(fovy / 2.0)
    return np.array([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (zfar + znear) / (znear - zfar), -1],
        [0, 0, (2 * zfar * znear) / (znear - zfar), 0],
    ], dtype=np.float32)

def look_at(eye, target, up):
    f = (target - eye); f = f / np.linalg.norm(f)
    s = np.cross(f, up); s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    M = np.identity(4, dtype=np.float32)
    M[0, :3] = s
    M[1, :3] = u
    M[2, :3] = -f
    T = np.identity(4, dtype=np.float32)
    T[:3, 3] = -eye
    return M @ T

def translate(x, y, z):
    M = np.identity(4, dtype=np.float32)
    M[:3, 3] = [x, y, z]
    return M

def scale(x, y, z):
    M = np.identity(4, dtype=np.float32)
    M[0, 0] = x; M[1, 1] = y; M[2, 2] = z
    return M

# --------------- Renderer ---------------
class Renderer:
    def __init__(self):
        pygame.init()
        pygame.display.set_mode(WINDOW_SIZE, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Virtual Factory 3D — Pygame + PyOpenGL")
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glClearColor(0.05, 0.07, 0.12, 1.0)
        self.program = create_program(VERT_SRC, FRAG_SRC)
        self.aPos = glGetAttribLocation(self.program, 'aPos')
        self.aNormal = glGetAttribLocation(self.program, 'aNormal')
        self.uMVP = glGetUniformLocation(self.program, 'uMVP')
        self.uModel = glGetUniformLocation(self.program, 'uModel')
        self.uNormalMat = glGetUniformLocation(self.program, 'uNormalMat')
        self.uColor = glGetUniformLocation(self.program, 'uColor')
        self.uViewPos = glGetUniformLocation(self.program, 'uViewPos')
        self.uLightDir = glGetUniformLocation(self.program, 'uLightDir')
        self.uLightColor = glGetUniformLocation(self.program, 'uLightColor')
        self.camera = Camera()
        self._init_geometry()
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 16)

    def _init_geometry(self):
        # Cube VBOs
        P, N = make_cube()
        self.vbo_pos = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_pos)
        glBufferData(GL_ARRAY_BUFFER, P.nbytes, P, GL_STATIC_DRAW)
        self.vbo_nrm = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_nrm)
        glBufferData(GL_ARRAY_BUFFER, N.nbytes, N, GL_STATIC_DRAW)
        self.vert_count = len(P)

    def draw_cube(self, MVP, model, color=(0.7, 0.7, 0.7)):
        glUseProgram(self.program)
        glUniformMatrix4fv(self.uMVP, 1, GL_FALSE, MVP)
        glUniformMatrix4fv(self.uModel, 1, GL_FALSE, model)
        normal_mat = np.linalg.inv(model[:3,:3]).T.astype(np.float32)
        glUniformMatrix3fv(self.uNormalMat, 1, GL_FALSE, normal_mat)
        glUniform3f(self.uColor, *color)

        view_pos = self.camera.pos()
        glUniform3f(self.uViewPos, float(view_pos[0]), float(view_pos[1]), float(view_pos[2]))
        glUniform3f(self.uLightDir, -0.7, -1.0, -0.4)
        glUniform3f(self.uLightColor, 1.0, 1.0, 1.0)

        glEnableVertexAttribArray(self.aPos)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_pos)
        glVertexAttribPointer(self.aPos, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(self.aNormal)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_nrm)
        glVertexAttribPointer(self.aNormal, 3, GL_FLOAT, GL_FALSE, 0, None)
        glDrawArrays(GL_TRIANGLES, 0, self.vert_count)
        glDisableVertexAttribArray(self.aPos)
        glDisableVertexAttribArray(self.aNormal)
        glUseProgram(0)

    def draw_text(self, surf, text, x, y):
        img = self.font.render(text, True, (230, 235, 245))
        surf.blit(img, (x, y))

    def run(self, snapshot_source):
        running = True
        yaw, pitch = 0.0, 0.0
        dragging = False
        last_mouse = (0, 0)

        while running:
            dt = self.clock.tick(60) / 1000.0
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    dragging = True; last_mouse = e.pos
                elif e.type == pygame.MOUSEBUTTONUP and e.button == 1:
                    dragging = False
                elif e.type == pygame.MOUSEMOTION and dragging:
                    dx, dy = e.pos[0]-last_mouse[0], e.pos[1]-last_mouse[1]
                    last_mouse = e.pos
                    self.camera.theta += dx * 0.005
                    self.camera.phi = max(0.1, min(1.5, self.camera.phi - dy * 0.005))
                elif e.type == pygame.MOUSEWHEEL:
                    self.camera.radius = max(6.0, min(80.0, self.camera.radius * (0.9 if e.y>0 else 1.1)))
            keys = pygame.key.get_pressed()
            if keys[pygame.K_ESCAPE]:
                running = False
            # WASD: move target on floor plane
            mv = np.array([0.0,0.0,0.0])
            if keys[pygame.K_w]: mv[2] -= 10*dt
            if keys[pygame.K_s]: mv[2] += 10*dt
            if keys[pygame.K_a]: mv[0] -= 10*dt
            if keys[pygame.K_d]: mv[0] += 10*dt
            if keys[pygame.K_q]: self.camera.target[1] += 5*dt
            if keys[pygame.K_e]: self.camera.target[1] -= 5*dt
            self.camera.target[:3] += mv

            # fetch snapshot
            snap = snapshot_source()

            # ---------- OpenGL frame ----------
            glViewport(0, 0, *WINDOW_SIZE)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            aspect = WINDOW_SIZE[0] / WINDOW_SIZE[1]
            P = perspective(math.radians(45.0), aspect, 0.1, 200.0)
            V = look_at(self.camera.pos(), self.camera.target, self.camera.up)

            # floor
            floor_model = scale(FLOOR_SIZE[0], 0.1, FLOOR_SIZE[1]) @ translate(0, -0.05, 0)
            MVP = (P @ V @ floor_model).astype(np.float32)
            self.draw_cube(MVP, floor_model.astype(np.float32), color=(0.12,0.16,0.22))

            # grid (simple strips along X and Z)
            for gx in range(-25, 26, 2):
                strip = scale(0.05, 0.02, 50.0) @ translate(gx, 0.0, 0)
                self.draw_cube((P@V@strip).astype(np.float32), strip.astype(np.float32), color=(0.18,0.22,0.30))
            for gz in range(-25, 26, 2):
                strip = scale(50.0, 0.02, 0.05) @ translate(0, 0.0, gz)
                self.draw_cube((P@V@strip).astype(np.float32), strip.astype(np.float32), color=(0.18,0.22,0.30))

            # stations 5x10
            idx = 0
            for r in range(GRID_ROWS):
                for c in range(GRID_COLS):
                    s = snap.stations[idx]
                    color = STATE_COLORS.get(s.state, (0.7,0.7,0.7))
                    x = (c - (GRID_COLS-1)/2.0) * STATION_SPACING_X
                    z = (r - (GRID_ROWS-1)/2.0) * STATION_SPACING_Z
                    model = translate(x, 0.75, z) @ scale(2.2, 1.5, 2.6)
                    self.draw_cube((P@V@model).astype(np.float32), model.astype(np.float32), color=color)
                    # buffer tower
                    if s.buffer>0:
                        b_model = translate(x+1.6, 0.2 + 0.15*s.buffer, z-1.1) @ scale(0.3, 0.3*s.buffer, 0.3)
                        self.draw_cube((P@V@b_model).astype(np.float32), b_model.astype(np.float32), color=(0.5,0.8,0.9))
                    idx += 1

            # conveyors (between columns)
            for r in range(GRID_ROWS):
                z = (r - (GRID_ROWS-1)/2.0) * STATION_SPACING_Z
                belt = translate(0, 0.2, z) @ scale(GRID_COLS*STATION_SPACING_X+1.5, 0.08, 0.6)
                self.draw_cube((P@V@belt).astype(np.float32), belt.astype(np.float32), color=(0.25,0.28,0.32))

            # AGVs
            for (x,y,z) in snap.agv:
                model = translate(x, 0.35, z) @ scale(1.0, 0.35, 0.7)
                self.draw_cube((P@V@model).astype(np.float32), model.astype(np.float32), color=(0.15,0.85,0.65))

            # swap buffers
            pygame.display.flip()

            # ---------- 2D HUD (pygame surface) ----------
            hud = pygame.display.get_surface()
            self.draw_text(hud, f"t={int(snap.t)}s  TH={snap.throughput}  WIP={snap.wipBusy}  Backlog={snap.backlog}", 12, 10)
            x0 = 12; y0 = 32
            inv_txt = "  ".join([f"{k}:{v}" for k,v in snap.inventory.items()])
            self.draw_text(hud, inv_txt, x0, y0)

        pygame.quit()

# --------------- Entrypoint ---------------

def main():
    renderer = Renderer()
    if USE_LIVE:
        live = LiveSimThread(); live.start()
        snapshot_source = lambda: (live.snapshot or make_demo_snapshot(time.time() % 1000.0))
    else:
        snapshot_source = lambda: make_demo_snapshot(time.time() % 1000.0)
    renderer.run(snapshot_source)

if __name__ == "__main__":
    main()
