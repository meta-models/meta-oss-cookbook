import type { MuseAnimation } from "../lib/agentPresentation";
import type { TranscriptEntry } from "../lib/transcript";
import type { MuseFace } from "./avatarFaces";

interface FaceRule {
  face: MuseFace;
  pattern: RegExp;
}

const sharedRules: FaceRule[] = [
  {
    face: "scared",
    pattern:
      /\b(afraid|danger|dangerous|fear|frighten|scared|terrified|worry|worried)\b/i,
  },
  {
    face: "sad",
    pattern: /\b(cry|grief|hurt|lonely|miss you|sad|sorry|unhappy|upset)\b/i,
  },
  {
    face: "angry",
    pattern: /\b(angry|annoyed|furious|hate|mad|outraged)\b/i,
  },
  {
    face: "shy",
    pattern: /\b(awkward|embarrassed|private|secret|shy)\b/i,
  },
  {
    face: "sleepy",
    pattern: /\b(exhausted|sleep|sleepy|tired|yawn)\b/i,
  },
];

const questionRules: FaceRule[] = [
  ...sharedRules,
  {
    face: "confused",
    pattern:
      /\b(confused|confusing|don't understand|doesn't make sense|what do you mean)\b/i,
  },
  {
    face: "suspicious",
    pattern: /\b(are you sure|prove|really true|seriously|trust|verify)\b/i,
  },
  {
    face: "surprised",
    pattern: /\b(amazing|no way|really|surprise|surprised|wow)\b/i,
  },
  {
    face: "unimpressed",
    pattern: /\b(boring|whatever|who cares)\b/i,
  },
  {
    face: "curious",
    pattern: /\?|\b(how|what|when|where|which|who|why)\b/i,
  },
];

const responseRules: FaceRule[] = [
  ...sharedRules,
  {
    face: "laughing",
    pattern: /\b(ha(?:ha)+|funny|joke|lol)\b/i,
  },
  {
    face: "proud",
    pattern: /\b(congratulations|proud|well done|you did it)\b/i,
  },
  {
    face: "excited",
    pattern:
      /!|\b(amazing|awesome|excellent|fantastic|great news|wonderful)\b/i,
  },
  {
    face: "confused",
    pattern: /\b(I don't know|I(?:'m| am) not sure|unclear|uncertain)\b/i,
  },
];

function matchFace(
  text: string,
  rules: FaceRule[],
  fallback: MuseFace,
): MuseFace {
  return rules.find((rule) => rule.pattern.test(text))?.face ?? fallback;
}

function latestEntry(entries: TranscriptEntry[], speaker: "user" | "agent") {
  for (let index = entries.length - 1; index >= 0; index -= 1) {
    const entry = entries[index];
    if (entry.speaker === speaker && entry.text.trim().length > 0) return entry;
  }
  return undefined;
}

export function defaultFaceForAnimation(animation: MuseAnimation): MuseFace {
  switch (animation) {
    case "listening":
    case "working":
      return "attentive";
    case "thinking":
      return "curious";
    case "happy":
      return "happy";
    case "idle":
      return "neutral";
  }
}

export function selectMuseFace(
  animation: MuseAnimation,
  entries: TranscriptEntry[],
): MuseFace {
  if (animation === "listening" || animation === "working") {
    return defaultFaceForAnimation(animation);
  }

  if (animation === "thinking") {
    const question = latestEntry(entries, "user");
    return question
      ? matchFace(question.text, questionRules, "curious")
      : "curious";
  }

  const response = latestEntry(entries, "agent");
  if (animation === "happy") {
    return response
      ? matchFace(response.text, responseRules, "happy")
      : "happy";
  }

  const latest = entries.at(-1);
  if (!latest) return "neutral";
  if (latest.speaker === "agent") {
    return matchFace(latest.text, responseRules, "happy");
  }
  return matchFace(latest.text, questionRules, "attentive");
}
