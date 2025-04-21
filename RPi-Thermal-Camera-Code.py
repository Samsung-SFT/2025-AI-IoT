import time
import numpy as np
import cv2
import board
import busio
from adafruit_mlx90640 import MLX90640

# Initialize I2C connection to MLX90640
i2c = busio.I2C(board.SCL, board.SDA)
mlx90640 = MLX90640(i2c)

# Set MLX90640 settings (optional)
mlx90640.refresh_rate = 0.5  # 0.5 Hz (2 seconds for each frame)
mlx90640.interleaved = True

# Function to create a constant color bar for temperature visualization
def create_colorbar():
    colorbar = np.zeros((256, 50, 3), dtype=np.uint8)
    for i in range(256):
        # Create color bar using the 'JET' colormap
        colorbar[i, :, :] = cv2.applyColorMap(np.array([[i]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0]
    return colorbar

# Initialize OpenCV window
cv2.namedWindow("Thermal Camera", cv2.WINDOW_NORMAL)

# Main loop
try:
    while True:
        # Capture thermal data
        thermal_data = read_thermal_data()

        # Check if data is valid
        if thermal_data is None:
            print("Error: No data received from MLX90640.")
            time.sleep(1)
            continue

        # Debugging: Check thermal data shape and range
        print(f"Thermal data shape: {thermal_data.shape}")
        print(f"Min/Max values: {thermal_data.min()}, {thermal_data.max()}")

        # Normalize the thermal data for visualization
        thermal_data_normalized = cv2.normalize(thermal_data, None, 0, 255, cv2.NORM_MINMAX)

        # Convert the normalized thermal data to a color map for visualization
        thermal_data_colormap = cv2.applyColorMap(thermal_data_normalized.astype(np.uint8), cv2.COLORMAP_JET)

        # Create a constant color bar
        colorbar = create_colorbar()

        # Stack the colorbar next to the thermal image for better display
        stacked_image = np.hstack((thermal_data_colormap, colorbar))

        # Show the thermal image with constant color bar
        cv2.imshow("Thermal Camera", stacked_image)

        # Break the loop if 'q' is pressed
        if (cv2.waitKey(1) & 0xFF) == ord('q'):  # Fixed bitwise operation
            break

        # Sleep to control refresh rate (based on MLX90640's refresh rate)
        time.sleep(1 / mlx90640.refresh_rate)

except KeyboardInterrupt:
    print("Exiting...")
finally:
    cv2.destroyAllWindows()



