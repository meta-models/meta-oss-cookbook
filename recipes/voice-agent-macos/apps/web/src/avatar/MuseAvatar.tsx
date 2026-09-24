import { useId } from "react";

import type { MuseAnimation } from "../lib/agentPresentation";
import { avatarFaces, type MuseFace } from "./avatarFaces";
import { defaultFaceForAnimation } from "./avatarExpression";

interface MuseAvatarProps {
  animation: MuseAnimation;
  reducedMotion: boolean;
  face?: MuseFace;
}

const animationLabels: Record<MuseAnimation, string> = {
  idle: "ready",
  listening: "listening",
  thinking: "thinking",
  working: "connecting",
  happy: "speaking",
};

const bodyPath =
  "M91.36 0.1C92.66 3.09 93.68 6.26 94.38 9.44C95.07 12.62 95.45 15.94 95.51 19.19C95.57 22.45 95.31 25.77 94.73 28.97C94.15 32.18 93.24 35.39 92.05 38.41C90.86 41.44 89.35 44.41 87.59 47.15C85.83 49.89 83.76 52.5 81.51 54.84C79.25 57.18 76.71 59.34 74.04 61.19C71.37 63.05 68.46 64.67 65.48 65.96C62.5 67.25 59.33 68.27 56.16 68.96C52.98 69.65 49.07 68.9 46.42 70.07C43.78 71.25 42.46 74.18 40.26 76.02C38.07 77.86 35.71 79.58 33.25 81.1C30.8 82.63 28.19 83.99 25.52 85.15C22.85 86.31 20.06 87.28 17.23 88.04C14.41 88.79 11.49 89.35 8.59 89.68C5.68 90.02 2.71 90.14 -0.22 90.04C-3.14 89.94 -6.09 89.62 -8.96 89.1C-11.83 88.58 -14.69 87.83 -17.44 86.9C-20.19 85.96 -22.89 84.81 -25.45 83.5C-28.02 82.19 -30.48 80.63 -32.83 79.02C-35.18 77.42 -36.81 74.75 -39.56 73.89C-42.31 73.02 -46.11 74.14 -49.35 73.82C-52.6 73.5 -55.89 72.87 -59.04 71.96C-62.2 71.04 -65.33 69.82 -68.28 68.33C-71.23 66.84 -74.09 65.05 -76.72 63.04C-79.36 61.03 -81.85 58.74 -84.07 56.27C-86.29 53.8 -88.32 51.08 -90.05 48.24C-91.78 45.4 -93.28 42.34 -94.45 39.23C-95.63 36.12 -96.53 32.84 -97.11 29.57C-97.7 26.29 -97.98 22.91 -97.95 19.59C-97.92 16.27 -97.57 12.9 -96.93 9.65C-96.29 6.41 -95.34 3.17 -94.12 0.1C-92.9 -2.96 -91.38 -5.95 -89.63 -8.72C-87.89 -11.5 -85.85 -14.14 -83.65 -16.53C-81.45 -18.92 -78.37 -20.87 -76.43 -23.07C-74.49 -25.27 -72.87 -27.22 -72.02 -29.71C-71.17 -32.2 -71.8 -35.27 -71.33 -38C-70.86 -40.73 -70.14 -43.48 -69.2 -46.1C-68.26 -48.73 -67.07 -51.32 -65.68 -53.75C-64.29 -56.18 -62.66 -58.53 -60.86 -60.68C-59.06 -62.84 -57.04 -64.87 -54.89 -66.67C-52.74 -68.48 -50.39 -70.12 -47.96 -71.51C-45.52 -72.91 -42.93 -74.11 -40.3 -75.06C-37.66 -76 -34.9 -76.72 -32.16 -77.19C-29.41 -77.66 -26.58 -77.88 -23.81 -77.87C-21.04 -77.85 -18.24 -77.58 -15.53 -77.08C-12.83 -76.59 -10.14 -75.84 -7.59 -74.9C-5.03 -73.96 -2.54 -72.78 -0.22 -71.44C2.11 -70.1 4.23 -68.03 6.36 -66.86C8.5 -65.68 10.3 -64.4 12.58 -64.39C14.86 -64.37 17.48 -66.21 20.03 -66.79C22.57 -67.36 25.23 -67.72 27.86 -67.83C30.49 -67.94 33.18 -67.81 35.8 -67.44C38.43 -67.07 41.07 -66.46 43.59 -65.61C46.11 -64.77 48.6 -63.67 50.94 -62.38C53.27 -61.08 55.53 -59.55 57.59 -57.84C59.66 -56.14 61.6 -54.21 63.32 -52.16C65.03 -50.11 66.59 -47.85 67.91 -45.52C69.22 -43.19 70.34 -40.69 71.21 -38.16C72.07 -35.63 72.72 -32.98 73.11 -30.34C73.5 -27.7 72.37 -24.73 73.55 -22.33C74.74 -19.93 78.05 -18.24 80.22 -15.93C82.39 -13.62 84.72 -11.14 86.58 -8.46C88.43 -5.79 90.06 -2.88 91.36 0.1Z";

export function MuseAvatar({
  animation,
  reducedMotion,
  face,
}: MuseAvatarProps) {
  const idPrefix = useId().replaceAll(":", "");
  const titleId = `${idPrefix}-title`;
  const descriptionId = `${idPrefix}-description`;
  const maskId = `${idPrefix}-mask`;
  const fillId = `${idPrefix}-fill`;
  const shadowId = `${idPrefix}-shadow`;
  const visibleAnimation = reducedMotion ? "idle" : animation;
  const visibleFace = face ?? defaultFaceForAnimation(animation);
  const faceDefinition = avatarFaces[visibleFace];

  return (
    <div
      className="muse-avatar"
      data-animation={visibleAnimation}
      data-face={visibleFace}
      data-reduced-motion={reducedMotion || undefined}
    >
      <svg
        className="muse-avatar-art"
        viewBox="-125 -125 250 250"
        role="img"
        aria-labelledby={titleId}
        aria-describedby={descriptionId}
      >
        <title id={titleId}>Muse Glimmer</title>
        <desc id={descriptionId}>
          An abstract blue voice companion showing the {visibleFace} expression,
          currently {animationLabels[animation]}.
        </desc>
        <defs>
          <mask
            id={maskId}
            maskUnits="userSpaceOnUse"
            x="-158"
            y="-158"
            width="316"
            height="316"
          >
            <path d={bodyPath} fill="#fff" />
            <g className="muse-face">
              <g transform={faceDefinition.left.transform}>
                <path
                  className="muse-eye-cutout muse-eye-left"
                  d={faceDefinition.left.path}
                  fill="#000"
                />
              </g>
              <g transform={faceDefinition.right.transform}>
                <path
                  className="muse-eye-cutout muse-eye-right"
                  d={faceDefinition.right.path}
                  fill="#000"
                />
              </g>
            </g>
          </mask>
          <radialGradient
            id={fillId}
            gradientUnits="userSpaceOnUse"
            cx="-50.56"
            cy="-63.2"
            r="244.9"
          >
            <stop offset="0%" stopColor="#b5d6f9" />
            <stop offset="32%" stopColor="#62a9f3" />
            <stop offset="72%" stopColor="#3b93f0" />
            <stop offset="100%" stopColor="#275c95" />
          </radialGradient>
          <filter id={shadowId} x="-35%" y="-35%" width="170%" height="180%">
            <feDropShadow
              dx="0"
              dy="10"
              stdDeviation="10"
              floodColor="#174e86"
              floodOpacity="0.24"
            />
          </filter>
        </defs>
        <g className="muse-character" filter={`url(#${shadowId})`}>
          <g mask={`url(#${maskId})`}>
            <rect
              x="-158"
              y="-158"
              width="316"
              height="316"
              fill={`url(#${fillId})`}
            />
          </g>
          <path className="muse-body" d={bodyPath} />
          <g className="muse-voice" aria-hidden="true">
            <path d="M-20 99v7" />
            <path d="M-10 96v13" />
            <path d="M0 93v19" />
            <path d="M10 96v13" />
            <path d="M20 99v7" />
          </g>
        </g>
      </svg>
    </div>
  );
}
