"""Generate synthetic test frames with ant-like blobs for pipeline testing."""

import cv2
import numpy as np
import os


def generate_frame(width=640, height=480, num_ants=8, ant_positions=None, seed=None):
    """Generate a single frame with dark ant-like blobs on a light surface."""
    if seed is not None:
        np.random.seed(seed)

    bg_color = np.random.randint(180, 210)
    frame = np.full((height, width, 3), bg_color, dtype=np.uint8)

    noise = np.random.randint(-10, 10, (height, width, 3), dtype=np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    if ant_positions is None:
        ant_positions = []
        for _ in range(num_ants):
            x = np.random.randint(30, width - 30)
            y = np.random.randint(30, height - 30)
            ant_positions.append((x, y))

    for (x, y) in ant_positions:
        ant_color = np.random.randint(20, 60)
        body_len = np.random.randint(8, 16)
        body_width = np.random.randint(3, 6)
        angle = np.random.randint(0, 360)

        cv2.ellipse(frame, (x, y), (body_len, body_width), angle,
                     0, 360, (ant_color, ant_color, ant_color), -1)

        head_offset_x = int(body_len * 0.8 * np.cos(np.radians(angle)))
        head_offset_y = int(body_len * 0.8 * np.sin(np.radians(angle)))
        head_radius = np.random.randint(2, 4)
        cv2.circle(frame, (x + head_offset_x, y + head_offset_y),
                   head_radius, (ant_color, ant_color, ant_color), -1)

    return frame, ant_positions


def generate_video_sequence(num_frames=60, width=640, height=480, num_ants=8):
    """Generate a sequence of frames with ants moving between frames."""
    positions = []
    for _ in range(num_ants):
        x = np.random.randint(50, width - 50)
        y = np.random.randint(50, height - 50)
        dx = np.random.uniform(-3, 3)
        dy = np.random.uniform(-3, 3)
        positions.append([x, y, dx, dy])

    frames = []
    all_positions = []

    for _ in range(num_frames):
        current = [(int(p[0]), int(p[1])) for p in positions]
        frame, _ = generate_frame(width, height, ant_positions=current)
        frames.append(frame)
        all_positions.append(current)

        for p in positions:
            p[0] += p[2] + np.random.uniform(-0.5, 0.5)
            p[1] += p[3] + np.random.uniform(-0.5, 0.5)
            p[0] = np.clip(p[0], 20, width - 20)
            p[1] = np.clip(p[1], 20, height - 20)
            if p[0] <= 20 or p[0] >= width - 20:
                p[2] *= -1
            if p[1] <= 20 or p[1] >= height - 20:
                p[3] *= -1

    return frames, all_positions


def save_test_data(output_dir="data/test"):
    """Save sample test images and a test video."""
    os.makedirs(output_dir, exist_ok=True)

    frame, positions = generate_frame(seed=42)
    cv2.imwrite(os.path.join(output_dir, "sample_frame.png"), frame)
    print(f"Saved sample_frame.png with {len(positions)} ants")

    frames, all_positions = generate_video_sequence(num_frames=90, num_ants=10)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(os.path.join(output_dir, "test_ants.mp4"),
                          fourcc, 15.0, (640, 480))
    for f in frames:
        out.write(f)
    out.release()
    print(f"Saved test_ants.mp4 ({len(frames)} frames, {len(all_positions[0])} ants)")

    return output_dir


if __name__ == "__main__":
    save_test_data()
