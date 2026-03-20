# **VSCode Extension Development Plan (TODO)**

## *Agentic Coding • Code Review • Modern Terminal • Multi‑Agent System • Context Management*

---

## **1. Quickstart Overview**

This VSCode extension will provide:

* ⚡ **AI Code Generation & Editing** across large codebases  
* 🧠 **Multi‑Agent Coding System** (Architect, Coder, Reviewer, Documentation Agent)  
* 🔍 **Rich Code Review Diffs** inside VSCode  
* 🖥️ **Modern Terminal Panel** with AI‑enhanced command understanding  
* 🗂️ **Context Manager** for files, URLs, images, repos  
* 🔗 **Integrations:** GitHub Actions, Slack, Linear, Local/Remote Agents  
* 🌐 **Universal Input Box** for natural‑language → code, commands, fixes  
* ⚙️ Configurable UI, keybindings, settings, and agent profiles  

---

## **2. Core Extension Architecture (TOP‑LEVEL TODO)**

### **2.1 Extension Scaffolding**
- [ ] Create VSCode extension scaffolding (TypeScript)
- [ ] Add Webview panels for prompts, agents, diffs, settings
- [ ] Add server/client architecture (`vscode-languageclient`)
- [ ] Add background worker process for multi‑agent runtime
- [ ] Add extension activation events
  - `"onStartupFinished"`
  - `"onCommand"`
  - `"workspaceContains"`

---

## **3. Key Features TODO (Modeled after Warp)**

### **3.1 Code (Advanced Code Generation)**
#### Detect coding opportunities
- [ ] Implement code‑intent detector (NL → coding classification)
- [ ] Parse workspace symbols for large‑repo awareness
- [ ] Use AST‑aware code writer for safe insert/update/remove
- [ ] Add multi‑step agent coding pipeline:
  - Architect Agent → Task Plan
  - Coding Agent → File Edits
  - Reviewer Agent → Diff Validation

#### Advanced Code Generation Flow
- [ ] Inline code edits (quick‑fix style)
- [ ] Whole‑file rewrite with diff preview
- [ ] Multi‑file refactor support
- [ ] Function extraction + auto unit‑test generation

### **3.2 Modern Terminal Panel**
- [ ] Custom Webview Terminal UI (not relying on VSCode built‑in)
- [ ] Multi‑line editor with completions
- [ ] AI command generation
- [ ] Enhanced SSH support (optional plugin)
- [ ] Block‑based terminal history (like Warp Blocks)
- [ ] Terminal‑to‑Agent connector
- [ ] Command classification:
  - Shell command
  - Git command
  - Natural language
  - Chat prompt

### **3.3 Agents (Multi‑Agent system)**
#### Agent Runtime
- [ ] Agent manager service
- [ ] Register agents with metadata (types, capabilities)
- [ ] Support parallel agent processes (threaded workers)

#### Built‑in Agents
- [ ] Architect Agent (break down tasks)
- [ ] Coding Agent (generate code)
- [ ] Debugging Agent
- [ ] Terminal Agent (commands)
- [ ] Reviewer Agent
- [ ] Documentation Agent
- [ ] Knowledge Agent (docs, URLs, images)

### **3.4 Agent Context Management**
- [ ] Add context sidebar panel (Files, Directories, Git diffs, Code selections, Images, Folders, URLs, Repo documentation)
- [ ] Add “Pin” + “Unpin” functionality
- [ ] Agent Context API:
  - Provide context to prompt builder
  - LLM‑aware trimming & chunking
  - Prioritization strategy
- [ ] Persistent saved contexts per project

### **3.5 Multi‑Agent Management Panel**
- [ ] “Running Agents” sidebar (like Warp’s Agents tab)
- [ ] Agent state machine: `waiting`, `running`, `needs_input`, `completed`
- [ ] Notifications when agent needs approval
- [ ] Kill/Restart agent controls
- [ ] Agent logs & timeline view

### **3.6 Universal Input Box (Single UX for Commands + Prompts)**
- [ ] Floating input box (like Warp Universal Input)
- [ ] Auto‑detect: Code task, Natural language, Shell command, File creation, Debug request
- [ ] Context‑enabled prompt building
- [ ] AI suggestions inline
- [ ] Accept/regen/expand options
- [ ] Keyboard‑first workflow (`⌘K` to open)

### **3.7 Code Review Diff UX**
- [ ] Custom diff renderer (Webview)
- [ ] Inline apply suggestion
- [ ] Agent conversation tied to diff
- [ ] Highlight agent‑created edits
- [ ] Undo/redo integrated with workspace edits
- [ ] Git patch export mode

### **3.8 Integrations**
- [ ] OAuth‑based account linking: GitHub, Linear, Slack
- [ ] Cloud Agent execution:
  - Trigger agents from GitHub Actions
  - Remote code modifications
  - CI‑driven agent suggestions
- [ ] Webhook‑based notifications

---

## **4. Developer Experience TODO**

### Settings
- [ ] Global + workspace‑level config
- [ ] LLM provider selection
- [ ] Temperature, max tokens, cost limiter
- [ ] Agent presets
- [ ] Appearance themes

### Keybindings
- [ ] `⌘K` → Universal Input
- [ ] `⌘Shift+A` → Open Agents
- [ ] `⌘Shift+D` → Open Diff Review Panel
- [ ] `⌘Shift+T` → AI Terminal

### Customizable Prompts
- [ ] User‑defined prompt templates
- [ ] Prompt library panel
- [ ] Multi‑language prompt packs (Bangla/English)

---

## **5. Contextual Intelligence TODO**
- [ ] Auto‑detect relevant files for prompt
- [ ] Codebase embeddings (local vector store)
- [ ] Semantic search “Find relevant context”
- [ ] Repo‑wide symbol graph
- [ ] Change‑aware caching

---

## **6. Performance TODO**
- [ ] LLM request batching
- [ ] Incremental context building
- [ ] Streaming token display
- [ ] Partial‑agent execution
- [ ] Async workers for heavy jobs
- [ ] Lazy‑loaded webviews
- [ ] LRU caching for embeddings

---

## **7. Testing & QA TODO**
- [ ] Unit tests (Jest)
- [ ] Integration tests (VSCode Test Runner)
- [ ] End‑to‑end Webview tests (Playwright)
- [ ] Offline‑mode testing
- [ ] Performance & latency benchmarking
- [ ] Agent correctness evaluation (task success scoring)

---

## **8. Release & CI/CD TODO**
- [ ] GitHub Actions CI
- [ ] Automated packaging + VSIX artifact
- [ ] Publish pipeline to VSCode Marketplace
- [ ] Canary/Beta release channels
- [ ] Telemetry (opt‑in only)
- [ ] Crash reporting system

---

## **9. Documentation TODO**
- [ ] Full docs site (Docusaurus)
- [ ] Developer guide
- [ ] API reference for agent development
- [ ] Recipes:
  - Refactoring large codebases
  - Debugging with agents
  - AI terminal workflows
  - Context management guide
- [ ] Video tutorials for new users

---

## **10. Community & Ecosystem TODO**
- [ ] Slack community for extension users
- [ ] Showcase example projects
- [ ] Extension plugin API (3rd‑party agents!)
- [ ] Pre‑built agent marketplace
- [ ] Feedback collector inside VSCode

---

## **11. Long‑Term Roadmap (Optional)**
- [ ] On‑device LLM support
- [ ] Local RAG server shipped inside extension
- [ ] Multi‑developer collaborative agent workflows
- [ ] Project‑wide task graph executor
- [ ] AI‑powered CI/CD analysis
- [ ] Voice input for Universal Input
- [ ] Vision model integration for design → code

---
