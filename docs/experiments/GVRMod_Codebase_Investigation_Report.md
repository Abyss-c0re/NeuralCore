# VRMod-x64 Codebase Investigation Report

## Executive Summary
This report documents a comprehensive investigation of the `vrmod-x64` Garry's Mod addon codebase using tree-sitter analysis tools. The investigation identified **61 Lua files** containing **1,112 symbols** (functions, methods, and class declarations) organized across a modular architecture supporting VR-specific gameplay mechanics, physics interactions, and player customization.

---

## 1. Project Structure Overview

### Root Files
- `addon.json` - GMod addon manifest defining metadata
- `README.md` - User documentation
- `LICENSE` - License terms
- `.github/FUNDING.yml` - Funding configuration

### Core Architecture (`lua/autorun/`)
- `vrmod_init.lua` - Main initialization entry point, loads core systems and registers the addon.

### API Layer (`lua/vrmod/api/`)
- `cl_api.lua` - Client-side API functions
- `sh_api.lua` - Shared API definitions
- `sv_api.lua` - Server-side API functions

### Module Organization
The codebase follows a clear module pattern with dedicated subdirectories:

| Category | Subdirectory | Purpose |
|----------|--------------|---------|
| **API** | `/api/` | Client/server/shared API abstractions |
| **Core** | `/core/` | Core initialization and system startup |
| **Input** | `/input/` | VR controller, button mapping, and input handling |
| **Network** | `/network/` | Server-client synchronization (collisions, weapons) |
| **Physics** | `/physics/` | Collision proxies and physics interactions |
| **Pickup** | `/pickup/` | Weapon pickup/replacement systems |
| **Player** | `/player/` | Player customization, character models, hand tracking |
| **UI** | `/ui/` | In-game menus, settings, HUD elements |
| **Utils** | `/utils/` | Helper utilities (math, rendering, trace, vehicles) |

---

## 2. Key Functional Modules

### 2.1 Core Systems
- `vrmod_init.lua`: Initializes the addon's global state (`g_VR`, `vrmod` tables), registers core systems, and sets up the module status tracking.
- `sh_startup.lua`: Handles server-side startup logic for shared systems.

### 2.2 Input & Controls
The `/input/` directory contains specialized modules for:
- SteamVR bindings (`cl_steamvr_bindings.lua`)
- Button mapping and custom actions (`sh_buttons.lua`, `cl_input.lua`)
- Specific VR interactions: climbing, doors, gliding, gravity manipulation, physgun, locomotion

### 2.3 Network Synchronization
- `sh_collisions_sync.lua`: Server-client collision data synchronization
- `sh_network.lua`: General network communication handling

### 2.4 Physics & Collision
- `sv_collision_proxies.lua`: Manages physics collision proxies for VR interactions
- Utility functions for trace operations and vehicle physics (`sh_trace.lua`, `sh_vehicles.lua`)

### 2.5 Weapon Systems
- `weapon_vrmod_empty.lua`: Implements the "Empty Hand" weapon model (arms viewmodel, no ammo)
- `sv_weaponreplacer.lua`: Server-side weapon replacement logic for pickup systems
- `vrmod_weps.lua` (inferred from context): Weapon utilities and management

### 2.6 Player Customization
- `cl_character.lua`, `cl_character_hands.lua`: Client-side character model handling
- `cl_foregrip.lua`: Foregrip attachment system
- `cl_laser_pointer.lua`: Laser pointer implementation
- `cl_viewmodeledit.lua`: Viewmodel editing tools

### 2.7 User Interface
- `cl_actioneditor.lua`: Custom action editor with VGUI-based UI for managing input actions
- `cl_settings.lua`, `cl_quickmenu.lua`: Settings and quick menu implementations
- `cl_worldtips.lua`: In-game tooltip system

### 2.8 Utilities
The `/utils/` directory provides helper functions for:
- Math operations (`sh_math.lua`)
- Rendering and beam effects (`cl_rendering.lua`)
- Frame management (`sh_frames.lua`)
- NPC to ragdoll conversion (`sh_npc2rag.lua`)
- System utilities (`sh_system.lua`)

---

## 3. Data Flow & Architecture Patterns

### 3.1 Global State Management
The codebase uses a hierarchical global state pattern:
```lua
g_VR = g_VR or {}  -- Primary global VR state table
vrmod = vrmod or {}  -- Secondary module-level state
```

This allows for clean separation of concerns and easy extension without polluting the global namespace.

### 3.2 Client/Server Separation
Functions follow clear client/server boundaries:
- `if SERVER then return end` - Server-side only functions
- `if CLIENT then return end` - Client-side only functions
- Shared logic marked appropriately in `/api/sh_*` files

### 3.3 VGUI-Based UI System
All user-facing interfaces are built using GMod's VGUI framework:
- `DFrame`, `DScrollPanel`, `DTextEntry`, `DCheckBox`, `DButton` components
- Dynamic UI creation with proper event handling (e.g., `OnClose`, `DoClick`)

### 3.4 Custom Action System
The action editor (`cl_actioneditor.lua`) demonstrates a sophisticated custom input system:
- Stores actions in `g_VR.CustomActions` table
- Persists to `vrmod/vrmod_custom_actions.txt` via JSON serialization
- Dynamically modifies `vrmod_action_manifest.txt` to integrate custom actions

---

## 4. Symbol Analysis (Tree-Sitter Results)

### Total Symbols Found: 1,112

#### Function Declarations & Signatures
Based on tree-sitter parsing, the codebase contains approximately 800-900 function declarations across 61 files. Key patterns include:

- **Utility Functions**: `vrmod.utils.CreateWorldModelVM`, `VRUtilLoadCustomActions`
- **UI Creation Functions**: Dynamic VGUI creation with proper cleanup (`OnClose`, `Remove`)
- **Event Handlers**: Input bindings, weapon events, network callbacks

#### Module-Level Structures
- `g_VR.CustomActions`: Custom action storage (dynamic array)
- `vrmod.status`: Boolean flags tracking module initialization state
- `vrmod.utils`: Utility functions namespace with client-side implementations

---

## 5. File-Specific Highlights

### Top-Level Initialization (`lua/autorun/vrmod_init.lua`)
```lua
vrmod = vrmod or {}
vrmod.status = {
    api = false,
    utils = false,
    core = false,
    network = false,
    input = false,
    player = false,
    physics = false,
    pickup = false,
    combat = false,
}
```
Provides status tracking for all major subsystems.

### API Layer (`lua/vrmod/api/`)
Contains the core API definitions for:
- Client-side operations (`cl_api.lua`)
- Server-side operations (`sv_api.lua`)
- Shared utilities (`sh_api.lua`)

### Custom Action Editor
The action editor demonstrates advanced VGUI patterns:
- Dynamic list rendering with `DScrollPanel`
- Text entry validation (alphanumeric + underscore)
- Persistent storage to disk
- Runtime manifest modification

---

## 6. Notable Implementations

### 6.1 Empty Hand Weapon
```lua
SWEP.PrintName = "Empty Hand"
SWEP.Slot = 0
SWEP.ViewModel = "models/weapons/c_arms.mdl"
SWEP.WorldModel = ""
SWEP.Primary.Ammo = "none"
```
A minimal weapon implementation providing an arms viewmodel without ammo or world model.

### 6.2 VR Pickup Lists Tool
- Category: `VRMod`
- Name: `VR Pickup Lists`
- Implements a client-side CPanel builder for pickup management

### 6.3 Network Synchronization
- Uses network strings (`util.AddNetworkString`) for server-to-client communication
- Collision data synchronization via dedicated module

---

## 7. Material & Asset References

- `materials/vrmod/` - Contains VRMod-specific material assets
- Notable: `tpbeam.vmt` (teleport beam visual effect)

---

## 8. Recommendations for Extension

Based on the architecture analysis:

1. **Maintain Client/Server Separation**: Continue using explicit `if SERVER/CLIENT` guards
2. **Leverage Global State**: Extend `g_VR` and `vrmod` tables for new features
3. **UI Consistency**: Follow existing VGUI patterns from `/ui/` directory
4. **Custom Actions**: Use the established action editor pattern for new input systems
5. **Persistence**: Store dynamic data to disk (`.txt`) with JSON serialization

---

## 9. Conclusion

The `vrmod-x64` codebase demonstrates a well-structured GMod addon with:
- Clear module boundaries and separation of concerns
- Robust client/server architecture
- Sophisticated VGUI-based UI systems
- Persistent custom action management
- Comprehensive VR-specific feature support (physics, locomotion, weapons)

The tree-sitter analysis confirmed **1,112 symbols** across **61 files**, indicating a mature codebase with good organization and maintainability practices.

---

*Report generated via tree-sitter codebase analysis tools.*
