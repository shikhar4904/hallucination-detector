import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext

from main import detect_hallucination


class HallucinationApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Hallucination Detector")
        self.root.geometry("900x620")

        self._build_ui()

    def _build_ui(self) -> None:
        title = tk.Label(
            self.root,
            text="Hallucination Detector (Main Pipeline Frontend)",
            font=("Segoe UI", 16, "bold"),
        )
        title.pack(pady=(14, 10))

        input_frame = tk.Frame(self.root)
        input_frame.pack(fill="x", padx=14)

        input_label = tk.Label(
            input_frame,
            text="Enter claim:",
            font=("Segoe UI", 11, "bold"),
        )
        input_label.pack(anchor="w")

        self.claim_box = scrolledtext.ScrolledText(
            input_frame,
            height=5,
            wrap=tk.WORD,
            font=("Segoe UI", 11),
        )
        self.claim_box.pack(fill="x", pady=(4, 8))
        self.claim_box.insert("1.0", "Albert Einstein was born in Germany in 1879.")

        button_frame = tk.Frame(input_frame)
        button_frame.pack(fill="x", pady=(0, 8))

        self.run_btn = tk.Button(
            button_frame,
            text="Analyze Claim",
            font=("Segoe UI", 10, "bold"),
            command=self._run_analysis,
            bg="#2E86DE",
            fg="white",
            padx=12,
            pady=6,
        )
        self.run_btn.pack(side="left")

        clear_btn = tk.Button(
            button_frame,
            text="Clear Output",
            font=("Segoe UI", 10),
            command=self._clear_output,
            padx=10,
            pady=6,
        )
        clear_btn.pack(side="left", padx=(10, 0))

        self.status_var = tk.StringVar(value="Ready")
        status_label = tk.Label(
            button_frame,
            textvariable=self.status_var,
            font=("Segoe UI", 10, "italic"),
            fg="#555555",
        )
        status_label.pack(side="right")

        output_label = tk.Label(
            self.root,
            text="Result:",
            font=("Segoe UI", 11, "bold"),
        )
        output_label.pack(anchor="w", padx=14)

        self.output_box = scrolledtext.ScrolledText(
            self.root,
            wrap=tk.WORD,
            font=("Consolas", 10),
        )
        self.output_box.pack(fill="both", expand=True, padx=14, pady=(4, 14))
        self.output_box.configure(state="disabled")

    def _run_analysis(self) -> None:
        claim = self.claim_box.get("1.0", tk.END).strip()
        if not claim:
            messagebox.showwarning("Input required", "Please enter a claim first.")
            return

        self.run_btn.configure(state="disabled")
        self.status_var.set("Analyzing...")
        self._write_output("Running pipeline, please wait...\n")

        # Run in background so GUI stays responsive while Wikipedia is fetched.
        thread = threading.Thread(target=self._analyze_claim, args=(claim,), daemon=True)
        thread.start()

    def _analyze_claim(self, claim: str) -> None:
        try:
            result = detect_hallucination(claim, verbose=False)
            text = self._format_result(result)
            self.root.after(0, lambda: self._finish_analysis(text, "Done"))
        except Exception as exc:
            error_text = f"Error while analyzing claim:\n{exc}"
            self.root.after(0, lambda: self._finish_analysis(error_text, "Failed"))

    def _finish_analysis(self, text: str, status: str) -> None:
        self._write_output(text)
        self.status_var.set(status)
        self.run_btn.configure(state="normal")

    def _clear_output(self) -> None:
        self.output_box.configure(state="normal")
        self.output_box.delete("1.0", tk.END)
        self.output_box.configure(state="disabled")
        self.status_var.set("Ready")

    def _write_output(self, text: str) -> None:
        self.output_box.configure(state="normal")
        self.output_box.delete("1.0", tk.END)
        self.output_box.insert(tk.END, text)
        self.output_box.configure(state="disabled")

    @staticmethod
    def _format_result(result: dict) -> str:
        verdict = result.get("verdict", "unknown").upper()
        main_entity = result.get("main_entity", "")
        hallucinated = result.get("hallucinated", [])
        evidence = result.get("evidence", [])

        lines = [
            "=" * 70,
            f"VERDICT     : {verdict}",
            f"MAIN ENTITY : {main_entity or '-'}",
            "=" * 70,
            "",
        ]

        if hallucinated:
            lines.append("FLAGGED DETAILS:")
            for item in hallucinated:
                lines.append(f"  - {item.get('detail', '')} ({item.get('type', '')})")
        else:
            lines.append("FLAGGED DETAILS: None")

        lines.append("")
        lines.append("EVIDENCE SENTENCES:")
        if evidence:
            for idx, sent in enumerate(evidence[:5], 1):
                lines.append(f"  {idx}. {sent}")
        else:
            lines.append("  No evidence found.")

        return "\n".join(lines)


def main() -> None:
    root = tk.Tk()
    app = HallucinationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
