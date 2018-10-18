

class SimulatorMap(object):
    """
        Encapsulates functions regarding the map and obstacles
        but no rendering etc. (So it can be run independently of the graphics.)
    """

    def __init__(self):
        # Map name, set in _load_map()
        self.map_name = None

        # Full map file path, set in _load_map()
        self.map_file_path = None

        # The parsed content of the map_file
        self.map_data = None

    def _load_map(self, map_name):
        """
        Load the map layout from a YAML file
        """

        # Store the map name
        self.map_name = map_name

        # Get the full map file path
        self.map_file_path = get_file_path('maps', map_name, 'yaml')

        logger.debug('loading map file "%s"' % self.map_file_path)

        with open(self.map_file_path, 'r') as f:
            self.map_data = yaml.load(f)

        tiles = self.map_data['tiles']
        assert len(tiles) > 0
        assert len(tiles[0]) > 0

        # Create the grid
        self.grid_height = len(tiles)
        self.grid_width = len(tiles[0])
        self.grid = [None] * self.grid_width * self.grid_height

        # We keep a separate list of drivable tiles
        self.drivable_tiles = []

        # For each row in the grid
        for j, row in enumerate(tiles):
            assert len(row) == self.grid_width, "each row of tiles must have the same length"

            # For each tile in this row
            for i, tile in enumerate(row):
                tile = tile.strip()

                if tile == 'empty':
                    continue

                if '/' in tile:
                    kind, orient = tile.split('/')
                    kind = kind.strip(' ')
                    orient = orient.strip(' ')
                    angle = ['S', 'E', 'N', 'W'].index(orient)
                    drivable = True
                elif '4' in tile:
                    kind = '4way'
                    angle = 2
                    drivable = True
                else:
                    kind = tile
                    angle = 0
                    drivable = False

                tile = {
                    'coords': (i, j),
                    'kind': kind,
                    'angle': angle,
                    'drivable': drivable
                }

                self._set_tile(i, j, tile)

                if drivable:
                    tile['curves'] = self._get_curve(i, j)
                    self.drivable_tiles.append(tile)

        self._load_objects(self.map_data)

        # Get the starting tile from the map, if specified
        self.start_tile = None
        if 'start_tile' in self.map_data:
            coords = self.map_data['start_tile']
            self.start_tile = self._get_tile(*coords)

    def _load_objects(self, map_data):
        # Create the objects array
        self.objects = []

        # The corners for every object, regardless if collidable or not
        self.object_corners = []

        # Arrays for checking collisions with N static objects
        # (Dynamic objects done separately)
        # (N x 2): Object position used in calculating reward
        self.collidable_centers = []

        # (N x 2 x 4): 4 corners - (x, z) - for object's boundbox
        self.collidable_corners = []

        # (N x 2 x 2): two 2D norms for each object (1 per axis of boundbox)
        self.collidable_norms = []

        # (N): Safety radius for object used in calculating reward
        self.collidable_safety_radii = []

        # For each object
        for obj_idx, desc in enumerate(map_data.get('objects', [])):
            kind = desc['kind']

            pos = desc['pos']
            x, z = pos[0:2]
            y = pos[2] if len(pos) == 3 else 0.0

            rotate = desc['rotate']
            optional = desc.get('optional', False)

            pos = ROAD_TILE_SIZE * np.array((x, y, z))

            # Load the mesh
            mesh = ObjMesh.get(kind)

            if 'height' in desc:
                scale = desc['height'] / mesh.max_coords[1]
            else:
                scale = desc['scale']
            assert not ('height' in desc and 'scale' in desc), "cannot specify both height and scale"

            static = desc.get('static', True)

            obj_desc = {
                'kind': kind,
                'mesh': mesh,
                'pos': pos,
                'scale': scale,
                'y_rot': rotate,
                'optional': optional,
                'static': static,
            }

            # obj = None
            if static:
                if kind == "trafficlight":
                    obj = TrafficLightObj(obj_desc, self.domain_rand, SAFETY_RAD_MULT)
                else:
                    obj = WorldObj(obj_desc, self.domain_rand, SAFETY_RAD_MULT)
            else:
                obj = DuckieObj(obj_desc, self.domain_rand, SAFETY_RAD_MULT, ROAD_TILE_SIZE)

            self.objects.append(obj)

            # Compute collision detection information

            # angle = rotate * (math.pi / 180)

            # Find drivable tiles object could intersect with
            possible_tiles = find_candidate_tiles(obj.obj_corners, ROAD_TILE_SIZE)

            # If the object intersects with a drivable tile
            if static and kind != "trafficlight" and self._collidable_object(
                    obj.obj_corners, obj.obj_norm, possible_tiles
            ):
                self.collidable_centers.append(pos)
                self.collidable_corners.append(obj.obj_corners.T)
                self.collidable_norms.append(obj.obj_norm)
                self.collidable_safety_radii.append(obj.safety_radius)

        # If there are collidable objects
        if len(self.collidable_corners) > 0:
            self.collidable_corners = np.stack(self.collidable_corners, axis=0)
            self.collidable_norms = np.stack(self.collidable_norms, axis=0)

            # Stack doesn't do anything if there's only one object,
            # So we add an extra dimension to avoid shape errors later
            if len(self.collidable_corners.shape) == 2:
                self.collidable_corners = self.collidable_corners[np.newaxis]
                self.collidable_norms = self.collidable_norms[np.newaxis]

        self.collidable_centers = np.array(self.collidable_centers)
        self.collidable_safety_radii = np.array(self.collidable_safety_radii)

   def _set_tile(self, i, j, tile):
        assert i >= 0 and i < self.grid_width
        assert j >= 0 and j < self.grid_height
        self.grid[j * self.grid_width + i] = tile

    def _get_tile(self, i, j):
        """
            Returns None if the duckiebot is not in a tile.
        """
        i = int(i)
        j = int(j)
        if i < 0 or i >= self.grid_width:
            return None
        if j < 0 or j >= self.grid_height:
            return None
        return self.grid[j * self.grid_width + i]




    def _drivable_pos(self, pos):
        """
        Check that the given (x,y,z) position is on a drivable tile
        """

        coords = self.get_grid_coords(pos)
        tile = self._get_tile(*coords)
        return tile is not None and tile['drivable']

    def _proximity_penalty2(self, pos, angle):
        """
        Calculates a 'safe driving penalty' (used as negative rew.)
        as described in Issue #24

        Describes the amount of overlap between the "safety circles" (circles
        that extend further out than BBoxes, giving an earlier collision 'signal'
        The number is max(0, prox.penalty), where a lower (more negative) penalty
        means that more of the circles are overlapping
        """

        pos = _actual_center(pos, angle)
        if len(self.collidable_centers) == 0:
            static_dist = 0

        # Find safety penalty w.r.t static obstacles
        else:
            d = np.linalg.norm(self.collidable_centers - pos, axis=1)

            if not safety_circle_intersection(d, AGENT_SAFETY_RAD, self.collidable_safety_radii):
                static_dist = 0
            else:
                static_dist = safety_circle_overlap(d, AGENT_SAFETY_RAD, self.collidable_safety_radii)

        total_safety_pen = static_dist
        for obj in self.objects:
            # Find safety penalty w.r.t dynamic obstacles
            total_safety_pen += obj.proximity(pos, AGENT_SAFETY_RAD)

        return total_safety_pen

    def _inconvenient_spawn(self, pos):
        """
        Check that agent spawn is not too close to any object
        """

        results = [np.linalg.norm(x.pos - pos) <
                   max(x.max_coords) * 0.5 * x.scale + MIN_SPAWN_OBJ_DIST
                   for x in self.objects if x.visible
                   ]
        return np.any(results)

    def _collision(self, agent_corners):
        """
        Tensor-based OBB Collision detection
        """

        # If there are no objects to collide against, stop
        if len(self.collidable_corners) == 0:
            return False

        # Generate the norms corresponding to each face of BB
        agent_norm = generate_norm(agent_corners)

        # Check collisions with static objects
        collision = intersects(
                agent_corners,
                self.collidable_corners,
                agent_norm,
                self.collidable_norms
        )

        if collision:
            return True

        # Check collisions with Dynamic Objects
        for obj in self.objects:
            if obj.check_collision(agent_corners, agent_norm):
                return True

        # No collision with any object
        return False

    def _valid_pose(self, pos, angle, safety_factor=1.0):
        """
            Check that the agent is in a valid pose

            safety_factor = minimum distance
        """

        # Compute the coordinates of the base of both wheels
        pos = _actual_center(pos, angle)
        f_vec = get_dir_vec(angle)
        r_vec = get_right_vec(angle)

        l_pos = pos - (safety_factor * 0.5 * ROBOT_WIDTH) * r_vec
        r_pos = pos + (safety_factor * 0.5 * ROBOT_WIDTH) * r_vec
        f_pos = pos + (safety_factor * 0.5 * ROBOT_LENGTH) * f_vec

        # Recompute the bounding boxes (BB) for the agent
        agent_corners = get_agent_corners(pos, angle)

        # Check that the center position and
        # both wheels are on drivable tiles and no collisions
        return (
                self._drivable_pos(pos) and
                self._drivable_pos(l_pos) and
                self._drivable_pos(r_pos) and
                self._drivable_pos(f_pos) and
                not self._collision(agent_corners)
        )


def _actual_center(pos, angle):
    """
    Calculate the position of the geometric center of the agent
    The value of self.cur_pos is the center of rotation.
    """

    dir_vec = get_dir_vec(angle)
    return pos + (CAMERA_FORWARD_DIST - (ROBOT_LENGTH / 2)) * dir_vec


def get_agent_corners(pos, angle):
    agent_corners = agent_boundbox(
            _actual_center(pos, angle),
            ROBOT_WIDTH,
            ROBOT_LENGTH,
            get_dir_vec(angle),
            get_right_vec(angle)
    )
    return agent_corners



def get_dir_vec(cur_angle):
    """
    Vector pointing in the direction the agent is looking
    """

    x = math.cos(cur_angle)
    z = -math.sin(cur_angle)
    return np.array([x, 0, z])




def get_right_vec(cur_angle):
    """
    Vector pointing to the right of the agent
    """

    x = math.sin(cur_angle)
    z = math.cos(cur_angle)
    return np.array([x, 0, z])
