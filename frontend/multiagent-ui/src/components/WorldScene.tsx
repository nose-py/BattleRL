import {
  useMemo,
  useRef,
  type ComponentType,
} from "react";

import { Canvas, useFrame } from "@react-three/fiber";

import {
  Billboard,
  Text,
  OrbitControls,
  RoundedBox,
} from "@react-three/drei";

import * as THREE from "three";

import {
  teamColor,
  samePoint,
} from "../lib/config";

import type {
  ActorState,
  Point,
  VisualEffect,
  WorldState,
} from "../types/simulation";

interface WorldSceneProps {
  state: WorldState;
  maxHealth: number;

  selectedCell: Point | null;
  onCellClick: (point: Point) => void;

  selectedAgentId: string;
  onSelectAgent: (id: string) => void;

  effects: VisualEffect[];
}

function tilePosition(
  point: Point,
  width: number,
  height: number
): [number, number, number] {
  return [
    point[0] - (width - 1) / 2,
    0,
    point[1] - (height - 1) / 2,
  ];
}

/* --------------------------------------------------
   Table Tile
-------------------------------------------------- */

function Tile({
  x,
  y,
  width,
  height,
  obstacle,
  selected,
  onClick,
}: {
  x: number;
  y: number;
  width: number;
  height: number;
  obstacle: boolean;
  selected: boolean;
  onClick: () => void;
}) {
  const [px, , pz] = tilePosition(
    [x, y],
    width,
    height
  );

  const baseColor = (x + y) % 2 === 0
    ? "#24354b"
    : "#293c51";

  return (
    <group position={[px, 0, pz]}>
      <RoundedBox
        args={[0.96, 0.18, 0.96]}
        radius={0.065}
        smoothness={4}
        position={[0, -0.14, 0]}
        receiveShadow
        onClick={(event) => {
          event.stopPropagation();
          onClick();
        }}
      >
        <meshStandardMaterial
          color={
            selected
              ? "#3b827e"
              : baseColor
          }
          metalness={0.12}
          roughness={0.75}
        />
      </RoundedBox>

      {selected && (
        <mesh position={[0, -0.038, 0]}>
          <boxGeometry args={[0.86, 0.018, 0.86]} />

          <meshBasicMaterial
            color="#6effd3"
            transparent
            opacity={0.24}
          />
        </mesh>
      )}

      {obstacle && (
        <group position={[0, 0.19, 0]}>
          <RoundedBox
            args={[0.72, 0.55, 0.72]}
            radius={0.11}
            smoothness={4}
            castShadow
            receiveShadow
          >
            <meshStandardMaterial
              color="#7a899b"
              roughness={0.88}
            />
          </RoundedBox>

          <mesh position={[0, 0.29, 0]}>
            <boxGeometry args={[0.42, 0.035, 0.42]} />

            <meshStandardMaterial
              color="#a6b3c2"
              roughness={0.8}
            />
          </mesh>
        </group>
      )}
    </group>
  );
}

/* --------------------------------------------------
   Goal Marker
-------------------------------------------------- */

function GoalMarker({
  position,
}: {
  position: [number, number, number];
}) {
  const ref = useRef<THREE.Group>(null);

  useFrame((state) => {
    if (!ref.current) return;

    ref.current.rotation.y =
      state.clock.elapsedTime * 0.65;

    ref.current.position.y =
      0.12 + Math.sin(state.clock.elapsedTime * 2) * 0.05;
  });

  return (
    <group position={position}>
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <torusGeometry args={[0.32, 0.028, 8, 48]} />

        <meshBasicMaterial color="#ffdb76" />
      </mesh>

      <group ref={ref}>
        <mesh>
          <octahedronGeometry args={[0.2, 0]} />

          <meshStandardMaterial
            color="#ffdb76"
            emissive="#ca8f28"
            emissiveIntensity={0.65}
            metalness={0.4}
            roughness={0.25}
          />
        </mesh>
      </group>

      <pointLight
        position={[0, 0.45, 0]}
        color="#f9c45c"
        intensity={1.3}
        distance={2.2}
      />
    </group>
  );
}

/* --------------------------------------------------
   3D Humanoid Character
-------------------------------------------------- */

function Humanoid({
  color,
  alive,
}: {
  color: string;
  alive: boolean;
}) {
  return (
    <group
      rotation={
        alive
          ? [0, 0, 0]
          : [0, 0, -Math.PI / 2.8]
      }
    >
      {/* Legs */}
      {[-0.1, 0.1].map((x) => (
        <mesh
          key={x}
          position={[x, 0.19, 0]}
          castShadow
        >
          <capsuleGeometry args={[0.065, 0.2, 4, 8]} />

          <meshStandardMaterial
            color="#aebfd0"
            roughness={0.6}
          />
        </mesh>
      ))}

      {/* Body */}
      <mesh position={[0, 0.52, 0]} castShadow>
        <capsuleGeometry args={[0.19, 0.29, 5, 12]} />

        <meshStandardMaterial
          color={color}
          metalness={0.35}
          roughness={0.4}
          emissive={color}
          emissiveIntensity={0.08}
        />
      </mesh>

      {/* Arms */}
      {[-1, 1].map((side) => (
        <mesh
          key={side}
          position={[side * 0.245, 0.49, 0]}
          rotation={[0, 0, side * 0.28]}
          castShadow
        >
          <capsuleGeometry args={[0.058, 0.25, 4, 8]} />

          <meshStandardMaterial
            color="#95a7ba"
            metalness={0.3}
            roughness={0.5}
          />
        </mesh>
      ))}

      {/* Head */}
      <mesh position={[0, 0.97, 0]} castShadow>
        <sphereGeometry args={[0.18, 20, 16]} />

        <meshStandardMaterial
          color="#e8f0f7"
          metalness={0.28}
          roughness={0.42}
        />
      </mesh>

      {/* Front Visor */}
      <mesh position={[0, 0.99, 0.145]}>
        <boxGeometry args={[0.25, 0.095, 0.075]} />

        <meshStandardMaterial
          color="#172c3b"
          emissive={color}
          emissiveIntensity={0.27}
          metalness={0.25}
          roughness={0.25}
        />
      </mesh>

      {/* Team Badge */}
      <mesh position={[0, 0.57, 0.185]}>
        <boxGeometry args={[0.13, 0.11, 0.022]} />

        <meshBasicMaterial color="#eafff8" />
      </mesh>
    </group>
  );
}

/* --------------------------------------------------
   Agent Label
-------------------------------------------------- */

function AgentLabel({
  id,
  team,
  health,
  maxHealth,
  selected,
}: {
  id: string;
  team: string;
  health: number;
  maxHealth: number;
  selected: boolean;
}) {
  const color = teamColor(team);

  const healthRatio = Math.max(
    0,
    Math.min(1, health / maxHealth)
  );

  const barWidth = 0.82;

  return (
    <Billboard
      position={[0, 1.5, 0]}
      follow
    >
      {/* Background of the label */}
      <mesh
        position={[0, 0, 0]}
        renderOrder={20}
      >
        <planeGeometry args={[1.05, 0.36]} />

        <meshBasicMaterial
          color={selected ? "#205149" : color}
          transparent
          opacity={0.94}
          depthTest={false}
          depthWrite={false}
        />
      </mesh>

      {/* Agent name */}
      <Text
        position={[0, 0.055, 0.03]}
        fontSize={0.13}
        maxWidth={0.95}
        anchorX="center"
        anchorY="middle"
        renderOrder={22}
      >
        {id}

        <meshBasicMaterial
          color="#000306"
          depthTest={false}
          depthWrite={false}
          toneMapped={false}
        />
      </Text>

      {/* Background of the health bar */}
      <mesh
        position={[0, -0.105, 0.04]}
        renderOrder={21}
      >
        <planeGeometry args={[barWidth, 0.055]} />

        <meshBasicMaterial
          color="#415468"
          transparent={true}
          opacity={1}
          depthTest={false}
          depthWrite={false}
          toneMapped={false}
        />
      </mesh>

      {/* Red health bar */}
      {healthRatio > 0 && (
        <mesh
          position={[
            -barWidth / 2 + (barWidth * healthRatio) / 2,
            -0.105,
            0.06,
          ]}
          renderOrder={22}
        >
          <planeGeometry
            args={[barWidth * healthRatio, 0.055]}
          />

          <meshBasicMaterial
            color="#B20A1A"
            transparent={true}
            opacity={1}
            depthTest={false}
            depthWrite={false}
            toneMapped={false}
          />
        </mesh>
      )}

      {/* Selected agent indicator */}
      {selected && (
        <mesh
          position={[-0.48, 0.075, 0.04]}
          renderOrder={23}
        >
          <circleGeometry args={[0.035, 12]} />

          <meshBasicMaterial
            color="#75ffe1"
            depthTest={false}
            depthWrite={false}
          />
        </mesh>
      )}
    </Billboard>
  );
}

/* --------------------------------------------------
   Animated Actor
-------------------------------------------------- */

function AnimatedActor({
  actor,
  width,
  height,
  maxHealth,
  selected,
  onSelect,
}: {
  actor: ActorState;
  width: number;
  height: number;
  maxHealth: number;

  selected: boolean;
  onSelect: () => void;
}) {
  const ref = useRef<THREE.Group>(null);

  const destination = useMemo(
    () => new THREE.Vector3(
      ...tilePosition(
        actor.position,
        width,
        height
      )
    ),
    [
      actor.position[0],
      actor.position[1],
      width,
      height,
    ]
  );

  const color = teamColor(actor.team);

  useFrame((_, delta) => {
    if (!ref.current) return;

    ref.current.position.lerp(
      destination,
      1 - Math.exp(-delta * 10)
    );
  });

  return (
    <group
      ref={ref}
      position={destination.toArray()}
      onClick={(event) => {
        event.stopPropagation();
        onSelect();
      }}
    >
      {/* Identification Base */}
      <mesh
        position={[0, -0.02, 0]}
        rotation={[-Math.PI / 2, 0, 0]}
      >
        <ringGeometry
          args={[
            selected ? 0.36 : 0.27,
            selected ? 0.45 : 0.34,
            32,
          ]}
        />

        <meshBasicMaterial
          color={selected ? "#75ffe1" : color}
          transparent
          opacity={selected ? 1 : actor.alive ? 0.85 : 0.2}
          side={THREE.DoubleSide}
        />
      </mesh>

      <group
        scale={actor.alive ? 1 : 0.8}
        position={actor.alive ? [0, 0, 0] : [0, -0.09, 0]}
      >
        <Humanoid
          color={actor.alive ? color : "#64748b"}
          alive={actor.alive}
        />
      </group>

      <AgentLabel
        id={actor.id}
        team={actor.team}
        health={actor.health}
        maxHealth={maxHealth}
        selected={selected}
      />
    </group>
  );
}

/* --------------------------------------------------
   Visual Effects
-------------------------------------------------- */

interface EffectProps {
  effect: VisualEffect;
  width: number;
  height: number;
}

function BeamEffect({
  effect,
  width,
  height,
}: EffectProps) {
  const material = useRef<THREE.MeshBasicMaterial>(null);
  const life = useRef(0);

  const from = tilePosition(
    effect.from ?? effect.to,
    width,
    height
  );

  const to = tilePosition(effect.to, width, height);

  const y = effect.kind === "shot" ? 0.62 : 0.08;

  const start = new THREE.Vector3(from[0], y, from[2]);
  const end = new THREE.Vector3(to[0], y, to[2]);

  const difference = end.clone().sub(start);
  const length = difference.length();

  const middle = start.clone().add(end).multiplyScalar(0.5);

  const quaternion = new THREE.Quaternion().setFromUnitVectors(
    new THREE.Vector3(0, 1, 0),
    difference.clone().normalize()
  );

  const color = effect.kind === "shot"
    ? "#ffb36d"
    : teamColor(effect.team ?? "");

  const duration = effect.kind === "shot"
    ? 0.42
    : 0.6;

  useFrame((_, delta) => {
    life.current += delta;

    if (material.current) {
      material.current.opacity = Math.max(
        0,
        1 - life.current / duration
      );
    }
  });

  if (length < 0.001) return null;

  return (
    <mesh
      position={middle.toArray()}
      quaternion={quaternion}
    >
      <cylinderGeometry
        args={[
          effect.kind === "shot" ? 0.045 : 0.02,
          effect.kind === "shot" ? 0.045 : 0.02,
          length,
          8,
        ]}
      />

      <meshBasicMaterial
        ref={material}
        color={color}
        transparent
        opacity={1}
        depthWrite={false}
      />
    </mesh>
  );
}

function PulseEffect({
  effect,
  width,
  height,
}: EffectProps) {
  const group = useRef<THREE.Group>(null);
  const material = useRef<THREE.MeshBasicMaterial>(null);
  const life = useRef(0);

  const position = tilePosition(
    effect.to,
    width,
    height
  );

  const settings: Record<
    string,
    {
      color: string;
      duration: number;
      scale: number;
    }
  > = {
    hit: {
      color: "#ff793f",
      duration: 0.55,
      scale: 1.4,
    },

    kill: {
      color: "#ff304f",
      duration: 0.85,
      scale: 2.4,
    },

    miss: {
      color: "#ffcf66",
      duration: 0.4,
      scale: 0.8,
    },

    invalid_move: {
      color: "#ff9855",
      duration: 0.45,
      scale: 0.7,
    },

    goal: {
      color: "#ffd86b",
      duration: 1.0,
      scale: 2.2,
    },

    first_goal: {
      color: "#55f5d0",
      duration: 1.35,
      scale: 3.1,
    },

    heal: {
      color: "#59efae",
      duration: 0.7,
      scale: 1.5,
    },
  };

  const config = settings[effect.kind] ?? {
    color: "#ffffff",
    duration: 0.6,
    scale: 1,
  };

  useFrame((_, delta) => {
    life.current += delta;

    const t = Math.min(
      1,
      life.current / config.duration
    );

    if (group.current) {
      group.current.scale.setScalar(
        0.35 + t * config.scale
      );
    }

    if (material.current) {
      material.current.opacity = 1 - t;
    }
  });

  return (
    <group
      ref={group}
      position={[
        position[0],
        0.25,
        position[2],
      ]}
    >
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <torusGeometry args={[0.32, 0.06, 8, 32]} />

        <meshBasicMaterial
          ref={material}
          color={config.color}
          transparent
          opacity={1}
          depthWrite={false}
          side={THREE.DoubleSide}
        />
      </mesh>

      {/* Núcleo luminoso del impacto */}
      {(effect.kind === "hit" ||
        effect.kind === "kill" ||
        effect.kind === "miss") && (
        <mesh position={[0, 0.12, 0]}>
          <sphereGeometry args={[0.11, 12, 12]} />

          <meshBasicMaterial
            color={config.color}
            transparent
            opacity={0.65}
          />
        </mesh>
      )}
    </group>
  );
}

/**
 * Registry of renderers.
 *
 * New effects here.
 */
const EFFECT_RENDERERS: Record<
  VisualEffect["kind"],
  ComponentType<EffectProps>
> = {
  // Trayectories
  move: BeamEffect,
  shot: BeamEffect,

  // Combat
  hit: PulseEffect,
  kill: PulseEffect,
  miss: PulseEffect,

  // Invalid moves
  invalid_move: PulseEffect,

  // Goals
  goal: PulseEffect,
  first_goal: PulseEffect,

  // Future healing
  heal: PulseEffect,
};

function EffectsLayer({
  effects,
  width,
  height,
}: {
  effects: VisualEffect[];
  width: number;
  height: number;
}) {
  return (
    <>
      {effects.map((effect) => {
        const Renderer = EFFECT_RENDERERS[effect.kind];

        return (
          <Renderer
            key={effect.id}
            effect={effect}
            width={width}
            height={height}
          />
        );
      })}
    </>
  );
}

/* --------------------------------------------------
   Complete Scene
-------------------------------------------------- */

function Battlefield({
  state,
  maxHealth,
  selectedCell,
  onCellClick,
  effects,
  selectedAgentId,
  onSelectAgent,
}: WorldSceneProps) {
  const { width, height } = state;

  const tiles = useMemo(() => {
    const result: Point[] = [];

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        result.push([x, y]);
      }
    }

    return result;
  }, [width, height]);

  return (
    <>
      <color attach="background" args={["#0d1828"]} />

      <ambientLight intensity={1.1} />

      <hemisphereLight
        color="#d4e8ff"
        groundColor="#192334"
        intensity={1.25}
      />

      <directionalLight
        position={[7, 12, 6]}
        intensity={2.5}
        castShadow
        shadow-mapSize={[2048, 2048]}
        shadow-camera-left={-18}
        shadow-camera-right={18}
        shadow-camera-top={18}
        shadow-camera-bottom={-18}
      />

      {/* Platform under the board */}
      <RoundedBox
        args={[width + 0.55, 0.45, height + 0.55]}
        position={[0, -0.47, 0]}
        radius={0.23}
        smoothness={5}
        receiveShadow
      >
        <meshStandardMaterial
          color="#152537"
          metalness={0.25}
          roughness={0.7}
        />
      </RoundedBox>

      {tiles.map(([x, y]) => {
        const obstacle = state.obstacles.some(
          (p) => p[0] === x && p[1] === y
        );

        return (
          <Tile
            key={`${x}-${y}`}
            x={x}
            y={y}
            width={width}
            height={height}
            obstacle={obstacle}
            selected={samePoint(selectedCell, [x, y])}
            onClick={() => onCellClick([x, y])}
          />
        );
      })}

      {state.goal && (
        <GoalMarker
          position={tilePosition(
            state.goal,
            width,
            height
          )}
        />
      )}

      {state.actors.map((actor) => (
        <AnimatedActor
          key={actor.id}
          actor={actor}
          width={width}
          height={height}
          maxHealth={maxHealth}
          selected={actor.id === selectedAgentId}
          onSelect={() => onSelectAgent(actor.id)}
        />
      ))}

      <EffectsLayer
        effects={effects}
        width={width}
        height={height}
      />

      <OrbitControls
        makeDefault
        enableDamping
        dampingFactor={0.08}
        minDistance={4}
        maxDistance={45}
        maxPolarAngle={Math.PI / 2.15}
      />
    </>
  );
}

export default function WorldScene(
  props: WorldSceneProps
) {
  const dimension = Math.max(
    props.state.width,
    props.state.height
  );

  return (
    <Canvas
      key={`${props.state.width}-${props.state.height}`}
      shadows
      dpr={[1, 1.75]}
      camera={{
        fov: 42,
        near: 0.1,
        far: 120,
        position: [
          dimension * 0.95,
          dimension * 1.28,
          dimension * 1.12,
        ],
      }}
      gl={{
        antialias: true,
        alpha: false,
      }}
    >
      <Battlefield {...props} />
    </Canvas>
  );
}