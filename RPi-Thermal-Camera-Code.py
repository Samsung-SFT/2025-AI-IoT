import time
import numpy as np
import cv2
import board
import busio
from adafruit_mlx90640 import MLX90640, RefreshRate  # <== import RefreshRate!

# Initialize I2C connection to MLX90640
i2c = busio.I2C(board.SCL, board.SDA)
mlx90640 = MLX90640(i2c)

# Set MLX90640 settings
mlx90640.refresh_rate = RefreshRate.REFRESH_0_5_HZ  # <== this line is fixed
mlx90640.interleaved = True

# Define the temperature range for the colorbar (adjust as needed)
TEMP_MIN = 20.0  # Minimum temperature in Celsius
TEMP_MAX = 40.0  # Maximum temperature in Celsius

# Function to create a constant color bar for temperature visualization
def create_colorbar():
    colorbar = np.zeros((256, 50, 3), dtype=np.uint8)
    for i in range(256):
        colorbar[i, :, :] = cv2.applyColorMap(np.array([[i]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0]
    return colorbar

# Create a static colorbar
colorbar = create_colorbar()

# Initialize OpenCV window
cv2.namedWindow("Thermal Camera", cv2.WINDOW_NORMAL)

# Function to read thermal data from MLX90640
def read_thermal_data():
    frame = np.zeros((24 * 32,), dtype=np.float32)
    mlx90640.getFrame(frame)
    return frame.reshape((24, 32))

# Main loop
try:
    while True:
        thermal_data = read_thermal_data()

        if thermal_data is None:
            print("Error: No data received from MLX90640.")
            time.sleep(1)
            continue

        print(f"Thermal data shape: {thermal_data.shape}")
        print(f"Min/Max values: {thermal_data.min()}, {thermal_data.max()}")

        thermal_data_normalized = np.clip((thermal_data - TEMP_MIN) / (TEMP_MAX - TEMP_MIN) * 255, 0, 255)

        thermal_data_colormap = cv2.applyColorMap(thermal_data_normalized.astype(np.uint8), cv2.COLORMAP_JET)

        stacked_image = np.hstack((thermal_data_colormap, colorbar))

        cv2.imshow("Thermal Camera", stacked_image)

        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

        time.sleep(2)  # 0.5Hz refresh means roughly 2 seconds between frames

except KeyboardInterrupt:
    print("Exiting...")
finally:
    cv2.destroyAllWindows()
