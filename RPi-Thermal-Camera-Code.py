import time
import numpy as np
import cv2
import board
import busio
from adafruit_mlx90640 import MLX90640, RefreshRate

# Initialize I2C connection to MLX90640
i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)  # Use 400kHz for faster speed
mlx90640 = MLX90640(i2c)

# Set MLX90640 settings
mlx90640.refresh_rate = RefreshRate.REFRESH_0_5_HZ  # Use enum instead of float

# Define temperature range for color mapping
TEMP_MIN = 20.0  # Celsius
TEMP_MAX = 40.0  # Celsius

# Function to create a static colorbar
def create_colorbar():
    colorbar = np.zeros((256, 50, 3), dtype=np.uint8)
    for i in range(256):
        color = cv2.applyColorMap(np.array([[i]], dtype=np.uint8), cv2.COLORMAP_JET)
        colorbar[i, :, :] = color[0, 0]
    return colorbar

# Function to read a frame from MLX90640
def read_thermal_data():
    frame = np.zeros((24 * 32,), dtype=np.float32)
    mlx90640.getFrame(frame)
    return frame.reshape((24, 32))

# Create colorbar once
colorbar = create_colorbar()

# Create OpenCV window
cv2.namedWindow("Thermal Camera", cv2.WINDOW_NORMAL)

try:
    while True:
        # Capture thermal data
        thermal_data = read_thermal_data()

        # Normalize the thermal data between 0-255
        normalized = np.interp(thermal_data, [TEMP_MIN, TEMP_MAX], [0, 255])
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)

        # Apply color map
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)

        # Stack colorbar beside the thermal image
        stacked = np.hstack((colored, colorbar))

        # Display
        cv2.imshow("Thermal Camera", stacked)

        # Exit if 'q' is pressed
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

        # Wait based on refresh rate
        time.sleep(2)  # Since 0.5Hz = 2 seconds per frame

except KeyboardInterrupt:
    print("Exiting...")

finally:
    cv2.destroyAllWindows()
