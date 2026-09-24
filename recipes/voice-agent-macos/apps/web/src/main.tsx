import "@fontsource/inter/latin-400.css";
import "@fontsource/inter/latin-500.css";
import "@fontsource/inter/latin-600.css";
import "@fontsource/inter/latin-700.css";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import App from "./App";
import "./app.css";

const root = document.getElementById("root");
if (!root) {
  throw new Error("The application root is missing.");
}

createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
