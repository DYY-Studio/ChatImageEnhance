import cv2
from core.orchestrator import ImageEnhancementOrchestrator
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="out.png")
    parser.add_argument("--prompt", default="自动增强")
    parser.add_argument("--trials", type=int, default=15)
    args = parser.parse_args()

    img = cv2.imread(args.input)
    orch = ImageEnhancementOrchestrator()
    best_rgb, params, df = orch.run(img, args.prompt, args.trials)
    cv2.imwrite(args.output, cv2.cvtColor(best_rgb, cv2.COLOR_RGB2BGR))
    print("Done. Best params:", params)

if __name__ == "__main__":
    main()