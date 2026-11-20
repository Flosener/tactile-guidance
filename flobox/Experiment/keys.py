import sys

if not sys.platform.startswith("linux"):
    from keyboard import is_pressed # Windows/macOS: unchanged

    def flush_input():
        if sys.platform.startswith("win"):
            import msvcrt
            while msvcrt.kbhit():
                msvcrt.getwch()
else:
    import selectors
    import termios
    import threading
    from evdev import InputDevice, ecodes, list_devices

    _pressed, _lock = set(), threading.Lock()
    _SPECIAL = {
        "left": [ecodes.KEY_LEFT], "right": [ecodes.KEY_RIGHT],
        "up": [ecodes.KEY_UP], "down": [ecodes.KEY_DOWN],
        "enter": [ecodes.KEY_ENTER, ecodes.KEY_KPENTER],
        "space": [ecodes.KEY_SPACE], "esc": [ecodes.KEY_ESC],
    }

    def _codes(name):
        n = name.lower() # 'G' and 'g' -> same key, like on Windows
        if n in _SPECIAL:
            return _SPECIAL[n]
        if len(n) == 1 and n.isalnum():
            return [ecodes.ecodes["KEY_" + n.upper()]]
        raise ValueError(f"Unsupported key name: {name!r}")

    def _find_keyboards():
        devs = []
        for path in list_devices():
            try:
                d = InputDevice(path)
            except PermissionError:
                continue
            caps = d.capabilities().get(ecodes.EV_KEY, [])
            if ecodes.KEY_A in caps and ecodes.KEY_ENTER in caps:
                devs.append(d)
        if not devs:
            raise RuntimeError(
                "No readable keyboard found in /dev/input. Add your user to the "
                "'input' group (sudo usermod -aG input $USER) and log in again.")
        return devs

    def _reader(devs):
        sel = selectors.DefaultSelector()
        for d in devs:
            sel.register(d, selectors.EVENT_READ)
        while True:
            for key, _ in sel.select():
                try:
                    for ev in key.fileobj.read():
                        if ev.type == ecodes.EV_KEY:
                            with _lock:
                                if ev.value:          # 1 = down, 2 = auto-repeat
                                    _pressed.add(ev.code)
                                else:                 # 0 = up
                                    _pressed.discard(ev.code)
                except OSError:                       # keyboard unplugged
                    sel.unregister(key.fileobj)

    threading.Thread(target=_reader, args=(_find_keyboards(),), daemon=True).start()

    def is_pressed(name):
        with _lock:
            return any(c in _pressed for c in _codes(name))

    def flush_input():
        """Throw away keys typed into the terminal (arrow keys, 's', ...)."""
        try:
            termios.tcflush(sys.stdin, termios.TCIFLUSH)
        except (termios.error, ValueError):
            pass