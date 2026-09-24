import executorchLogo from "../assets/et-logo.png";

export function RuntimeBadge() {
  return (
    <div className="runtime-badge" aria-label="Running on ExecuTorch">
      <img className="runtime-logo" src={executorchLogo} alt="" />
      <span>
        Running on <strong>ExecuTorch</strong>
      </span>
    </div>
  );
}
