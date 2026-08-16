import { useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { Float, Line, Sparkles } from '@react-three/drei';
import * as THREE from 'three';

import { AGENT_HEX } from '@/lib/agents';
import { FLOW_ROWS } from '@/lib/flow';
import type { AgentId } from '@/types/analysis';

/**
 * The engine, as a constellation.
 *
 * Not a decorative particle field — the twelve nodes are the twelve real
 * agents (`AGENT_HEX`), grouped into the same four pipeline tiers the
 * dashboard's own agent graph uses (`FLOW_ROWS`: orchestration, perception,
 * extraction, synthesis). The names and colours a visitor sees here are the
 * same ones they will see running for real one screen later.
 *
 * What this is *not* claiming: the connecting lines are a tier-adjacency
 * sketch, not the literal dependency edges `/api/agents` reports. Precision
 * about wiring belongs to the dashboard's AgentFlow graph, which reads the
 * roster directly — this is atmosphere built from real names, not a second
 * copy of a diagram that could drift from the source of truth.
 */

interface Node3D {
  id: AgentId;
  row: number;
  position: [number, number, number];
}

const ROW_RADIUS = [0, 2.6, 3.9, 3.1];
const ROW_Z = [1.2, -0.6, -2.8, -5.2];
const ROW_TILT = [0, 0.15, -0.1, 0.2];

function buildNodes(): Node3D[] {
  const nodes: Node3D[] = [];
  FLOW_ROWS.forEach((row, rowIndex) => {
    const radius = ROW_RADIUS[rowIndex];
    if (radius === 0) {
      nodes.push({ id: row[0], row: rowIndex, position: [0, 0, ROW_Z[rowIndex]] });
      return;
    }
    row.forEach((id, col) => {
      const angle =
        (col / row.length) * Math.PI * 2 + rowIndex * 0.7 + ROW_TILT[rowIndex];
      nodes.push({
        id,
        row: rowIndex,
        position: [
          Math.cos(angle) * radius,
          Math.sin(angle) * radius * 0.62,
          ROW_Z[rowIndex],
        ],
      });
    });
  });
  return nodes;
}

/** Nearest-angle pairing between adjacent tiers — a readable sketch of a
 * layered pipeline, not a claim about the real dependency graph. */
function buildEdges(nodes: Node3D[]): [Node3D, Node3D][] {
  const edges: [Node3D, Node3D][] = [];
  for (let row = 0; row < FLOW_ROWS.length - 1; row++) {
    const from = nodes.filter((n) => n.row === row);
    const to = nodes.filter((n) => n.row === row + 1);
    to.forEach((child) => {
      const parent = from.reduce((best, candidate) => {
        const d = (a: Node3D) =>
          (a.position[0] - child.position[0]) ** 2 +
          (a.position[1] - child.position[1]) ** 2;
        return d(candidate) < d(best) ? candidate : best;
      }, from[0]);
      if (parent) edges.push([parent, child]);
    });
  }
  return edges;
}

function AgentNode({ node, reduced }: { node: Node3D; reduced: boolean }) {
  const hex = AGENT_HEX[node.id];
  const size = node.row === 0 ? 0.26 : 0.13;

  const core = (
    <group position={node.position}>
      {/* Glow halo — additive-blended, larger, low opacity. The cheap way to
          fake bloom without a postprocessing pass this project doesn't ship. */}
      <mesh scale={2.4}>
        <sphereGeometry args={[size, 16, 16]} />
        <meshBasicMaterial
          color={hex}
          transparent
          opacity={0.16}
          depthWrite={false}
          blending={THREE.AdditiveBlending}
        />
      </mesh>
      <mesh>
        <icosahedronGeometry args={[size, node.row === 0 ? 1 : 0]} />
        <meshStandardMaterial
          color={hex}
          emissive={hex}
          emissiveIntensity={node.row === 0 ? 2.2 : 1.5}
          roughness={0.35}
          metalness={0.4}
        />
      </mesh>
    </group>
  );

  if (reduced) return core;

  return (
    <Float speed={1.3} rotationIntensity={0.35} floatIntensity={0.55}>
      {core}
    </Float>
  );
}

export function AgentConstellation({ reducedMotion }: { reducedMotion: boolean }) {
  const nodes = useMemo(buildNodes, []);
  const edges = useMemo(() => buildEdges(nodes), [nodes]);

  const group = useRef<THREE.Group>(null);
  const mouse = useRef({ x: 0, y: 0 });

  useFrame((state, delta) => {
    if (!group.current) return;

    if (!reducedMotion) {
      group.current.rotation.y += delta * 0.045;
      // Frame-rate independent damping toward the pointer — smoother than a
      // fixed-factor lerp, and it does not accelerate on a slow frame.
      mouse.current.x = THREE.MathUtils.damp(mouse.current.x, state.pointer.x, 4, delta);
      mouse.current.y = THREE.MathUtils.damp(mouse.current.y, state.pointer.y, 4, delta);
      group.current.rotation.x = mouse.current.y * -0.18;
      group.current.rotation.z = mouse.current.x * 0.08;
    }
  });

  return (
    <group ref={group}>
      {edges.map(([from, to], i) => (
        <Line
          key={i}
          points={[from.position, to.position]}
          color="#2A3A55"
          lineWidth={1}
          transparent
          opacity={0.4}
          dashed
          dashSize={0.14}
          gapSize={0.1}
        />
      ))}
      {nodes.map((node) => (
        <AgentNode key={node.id} node={node} reduced={reducedMotion} />
      ))}
      {!reducedMotion && (
        <Sparkles
          count={120}
          scale={[16, 9, 12]}
          size={1.3}
          speed={0.22}
          color="#3A4A6A"
          opacity={0.45}
        />
      )}
    </group>
  );
}
