# TypeScriptMate Continue Extension Setup

This guide will help you set up the Continue extension with TypeScriptMate.

## Installation

1. Open VS Code
2. Go to the Extensions view (Ctrl+Shift+X / Cmd+Shift+X)
3. Search for "Continue"
4. Click "Install" on the Continue extension by Continue AI

## Configuration

1. Create a new file called `config.yaml` in your project root directory
2. Copy the configuration to the Continue extension directory:

```bash
# For macOS
cp continue.json ~/Library/Application\ Support/Continue/config.json

# For Linux
cp continue.json ~/.config/Continue/config.json

# For Windows (PowerShell)
Copy-Item continue.json "$env:APPDATA\Continue\config.json"
```

3. Save the file
4. Copy the configuration to the Continue extension directory:

```bash
# For macOS
cp continue.json ~/Library/Application\ Support/Continue/config.json

# For Linux
cp continue.json ~/.config/Continue/config.json

# For Windows (PowerShell)
Copy-Item continue.json "$env:APPDATA\Continue\config.json"
```

## Usage

- Press `Cmd+P` (Mac) or `Ctrl+P` (Windows/Linux) to toggle Continue: Toggle Autocomplete Enabled
- The model will respond with code completions and suggestions
