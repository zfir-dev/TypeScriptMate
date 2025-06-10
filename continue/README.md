# TypeScriptMate Continue Extension Setup

This guide will help you set up the Continue extension with TypeScriptMate.

## Installation

1. Open VS Code
2. Go to the Extensions view (Ctrl+Shift+X / Cmd+Shift+X)
3. Search for "Continue"
4. Click "Install" on the Continue extension by Continue AI
   - Direct link: [Continue - VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=Continue.continue)

## Configuration

1. Create a new file called `config.yaml` in your project root directory
2. Copy the configuration to the Continue extension directory:

```bash
cp -f config.yaml ~/.continue/config.yaml
```

## Usage

- Press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows/Linux) to open the Command Palette
- Type "Continue: Toggle Autocomplete Enabled" and press Enter
- The model will respond with code completions and suggestions

## Settings
- Press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows/Linux) to open the Command Palette
- Type "Continue: Open Settings" and press Enter
- Find these settings:
- - Autocomplete Timeout (ms): Set it to `400`
- - Autocomplete Debounce (ms): Set it to `500`
- - Disable autocomplete in files: Set it to `**/*, !**/*.ts, !**/*.tsx`

