import base64

# Read and encode the icon file
with open("ICON.ico", "rb") as icon_file:
    encoded_icon = base64.b64encode(icon_file.read()).decode("utf-8")

# Write the encoded data to icon.py
with open("icon.py", "w") as py_file:
    py_file.write("icon_data = \"\"\"" + encoded_icon + "\"\"\"")
