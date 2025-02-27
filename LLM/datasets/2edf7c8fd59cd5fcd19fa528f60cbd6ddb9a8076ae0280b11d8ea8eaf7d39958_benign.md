### pstree

```json
  {
    "Audit": "\\Device\\HarddiskVolume3\\Windows\\System32\\winlogon.exe",
    "Cmd": "winlogon.exe",
    "CreateTime": "2024-11-19T19:42:02",
    "ExitTime": null,
    "Handles": null,
    "ImageFileName": "winlogon.exe",
    "Offset(V)": 204562857664640,
    "PID": 672,
    "PPID": 564,
    "Path": "C:\\Windows\\system32\\winlogon.exe",
    "SessionId": 1,
    "Threads": 3,
    "Wow64": false,
    "__children": [
      {
        "Audit": "\\Device\\HarddiskVolume3\\Windows\\System32\\userinit.exe",
        "Cmd": null,
        "CreateTime": "2024-11-19T19:42:06",
        "ExitTime": "2024-11-19T19:42:29",
        "Handles": null,
        "ImageFileName": "userinit.exe",
        "Offset(V)": 204562884797248,
        "PID": 4928,
        "PPID": 672,
        "Path": null,
        "SessionId": 1,
        "Threads": 0,
        "Wow64": false,
        "__children": [
          {
            "Audit": "\\Device\\HarddiskVolume3\\Windows\\explorer.exe",
            "Cmd": "C:\\Windows\\Explorer.EXE",
            "CreateTime": "2024-11-19T19:42:06",
            "ExitTime": null,
            "Handles": null,
            "ImageFileName": "explorer.exe",
            "Offset(V)": 204562885165888,
            "PID": 4988,
            "PPID": 4928,
            "Path": "C:\\Windows\\Explorer.EXE",
            "SessionId": 1,
            "Threads": 65,
            "Wow64": false,
            "__children": [
              {
                "Audit": "\\Device\\HarddiskVolume3\\Users\\BA-LK\\Documents\\2edf7c8fd59cd5fcd19fa528f60cbd6ddb9a8076ae0280b11d8ea8eaf7d39958.exe",
                "Cmd": "\"C:\\Users\\BA-LK\\Documents\\2edf7c8fd59cd5fcd19fa528f60cbd6ddb9a8076ae0280b11d8ea8eaf7d39958.exe\" ",
                "CreateTime": "2024-11-19T19:43:22",
                "ExitTime": null,
                "Handles": null,
                "ImageFileName": "2edf7c8fd59cd5",
                "Offset(V)": 204562881626240,
                "PID": 4832,
                "PPID": 4988,
                "Path": "C:\\Users\\BA-LK\\Documents\\2edf7c8fd59cd5fcd19fa528f60cbd6ddb9a8076ae0280b11d8ea8eaf7d39958.exe",
                "SessionId": 1,
                "Threads": 1,
                "Wow64": true,
                "__children": []
              },
              {
                "Audit": "\\Device\\HarddiskVolume3\\Windows\\System32\\SecurityHealthSystray.exe",
                "Cmd": "\"C:\\Windows\\System32\\SecurityHealthSystray.exe\" ",
                "CreateTime": "2024-11-19T19:42:19",
                "ExitTime": null,
                "Handles": null,
                "ImageFileName": "SecurityHealth",
                "Offset(V)": 204562893570176,
                "PID": 6668,
                "PPID": 4988,
                "Path": "C:\\Windows\\System32\\SecurityHealthSystray.exe",
                "SessionId": 1,
                "Threads": 5,
                "Wow64": false,
                "__children": []
              }
            ]
          }
        ]
      },
      {
        "Audit": "\\Device\\HarddiskVolume3\\Windows\\System32\\fontdrvhost.exe",
        "Cmd": "\"fontdrvhost.exe\"",
        "CreateTime": "2024-11-19T19:42:02",
        "ExitTime": null,
        "Handles": null,
        "ImageFileName": "fontdrvhost.ex",
        "Offset(V)": 204562858078528,
        "PID": 884,
        "PPID": 672,
        "Path": "C:\\Windows\\system32\\fontdrvhost.exe",
        "SessionId": 1,
        "Threads": 5,
        "Wow64": false,
        "__children": []
      },
      {
        "Audit": "\\Device\\HarddiskVolume3\\Windows\\System32\\dwm.exe",
        "Cmd": "\"dwm.exe\"",
        "CreateTime": "2024-11-19T19:42:03",
        "ExitTime": null,
        "Handles": null,
        "ImageFileName": "dwm.exe",
        "Offset(V)": 204562867351680,
        "PID": 1364,
        "PPID": 672,
        "Path": "C:\\Windows\\system32\\dwm.exe",
        "SessionId": 1,
        "Threads": 15,
        "Wow64": false,
        "__children": []
      }
    ]
  },
```

## malfind

```json
keine Results
```

### netscan

```json
keine Results
```

### pslist

```json
  {
    "CreateTime": "2024-11-19T19:43:22",
    "ExitTime": null,
    "File output": "Disabled",
    "Handles": null,
    "ImageFileName": "2edf7c8fd59cd5",
    "Offset(V)": 204562881626240,
    "PID": 4832,
    "PPID": 4988,
    "SessionId": 1,
    "Threads": 1,
    "Wow64": true,
    "__children": []
  },
  {
    "CreateTime": "2024-11-19T19:42:19",
    "ExitTime": null,
    "File output": "Disabled",
    "Handles": null,
    "ImageFileName": "SecurityHealth",
    "Offset(V)": 204562893570176,
    "PID": 6668,
    "PPID": 4988,
    "SessionId": 1,
    "Threads": 5,
    "Wow64": false,
    "__children": []
  },
  {
    "CreateTime": "2024-11-19T19:42:06",
    "ExitTime": null,
    "File output": "Disabled",
    "Handles": null,
    "ImageFileName": "explorer.exe",
    "Offset(V)": 204562885165888,
    "PID": 4988,
    "PPID": 4928,
    "SessionId": 1,
    "Threads": 65,
    "Wow64": false,
    "__children": []
  },
  {
    "CreateTime": "2024-11-19T19:42:06",
    "ExitTime": "2024-11-19T19:42:29",
    "File output": "Disabled",
    "Handles": null,
    "ImageFileName": "userinit.exe",
    "Offset(V)": 204562884797248,
    "PID": 4928,
    "PPID": 672,
    "SessionId": 1,
    "Threads": 0,
    "Wow64": false,
    "__children": []
  },
  {
    "CreateTime": "2024-11-19T19:42:03",
    "ExitTime": null,
    "File output": "Disabled",
    "Handles": null,
    "ImageFileName": "dwm.exe",
    "Offset(V)": 204562867351680,
    "PID": 1364,
    "PPID": 672,
    "SessionId": 1,
    "Threads": 15,
    "Wow64": false,
    "__children": []
  },
  {
    "CreateTime": "2024-11-19T19:42:02",
    "ExitTime": null,
    "File output": "Disabled",
    "Handles": null,
    "ImageFileName": "fontdrvhost.ex",
    "Offset(V)": 204562858078528,
    "PID": 884,
    "PPID": 672,
    "SessionId": 1,
    "Threads": 5,
    "Wow64": false,
    "__children": []
  },
  {
    "CreateTime": "2024-11-19T19:42:02",
    "ExitTime": null,
    "File output": "Disabled",
    "Handles": null,
    "ImageFileName": "winlogon.exe",
    "Offset(V)": 204562857664640,
    "PID": 672,
    "PPID": 564,
    "SessionId": 1,
    "Threads": 3,
    "Wow64": false,
    "__children": []
  },
```

### psscan

```json 
  {
    "CreateTime": "2024-11-19T19:43:22",
    "ExitTime": null,
    "File output": "Disabled",
    "Handles": null,
    "ImageFileName": "2edf7c8fd59cd5",
    "Offset(V)": 204562881626240,
    "PID": 4832,
    "PPID": 4988,
    "SessionId": 1,
    "Threads": 1,
    "Wow64": true,
    "__children": []
  },  
  {
    "CreateTime": "2024-11-19T19:42:19",
    "ExitTime": null,
    "File output": "Disabled",
    "Handles": null,
    "ImageFileName": "SecurityHealth",
    "Offset(V)": 204562893570176,
    "PID": 6668,
    "PPID": 4988,
    "SessionId": 1,
    "Threads": 5,
    "Wow64": false,
    "__children": []
  },
  {
    "CreateTime": "2024-11-19T19:42:06",
    "ExitTime": null,
    "File output": "Disabled",
    "Handles": null,
    "ImageFileName": "explorer.exe",
    "Offset(V)": 204562885165888,
    "PID": 4988,
    "PPID": 4928,
    "SessionId": 1,
    "Threads": 65,
    "Wow64": false,
    "__children": []
  },
  {
    "CreateTime": "2024-11-19T19:42:06",
    "ExitTime": "2024-11-19T19:42:29",
    "File output": "Disabled",
    "Handles": null,
    "ImageFileName": "userinit.exe",
    "Offset(V)": 204562884797248,
    "PID": 4928,
    "PPID": 672,
    "SessionId": 1,
    "Threads": 0,
    "Wow64": false,
    "__children": []
  },
  {
    "CreateTime": "2024-11-19T19:42:03",
    "ExitTime": null,
    "File output": "Disabled",
    "Handles": null,
    "ImageFileName": "dwm.exe",
    "Offset(V)": 204562867351680,
    "PID": 1364,
    "PPID": 672,
    "SessionId": 1,
    "Threads": 15,
    "Wow64": false,
    "__children": []
  },
  {
    "CreateTime": "2024-11-19T19:42:02",
    "ExitTime": null,
    "File output": "Disabled",
    "Handles": null,
    "ImageFileName": "fontdrvhost.ex",
    "Offset(V)": 204562858078528,
    "PID": 884,
    "PPID": 672,
    "SessionId": 1,
    "Threads": 5,
    "Wow64": false,
    "__children": []
  },
  {
    "CreateTime": "2024-11-19T19:42:02",
    "ExitTime": null,
    "File output": "Disabled",
    "Handles": null,
    "ImageFileName": "winlogon.exe",
    "Offset(V)": 204562857664640,
    "PID": 672,
    "PPID": 564,
    "SessionId": 1,
    "Threads": 3,
    "Wow64": false,
    "__children": []
  },
```

672 -->
    884 -->
    1364 --> 
    4928  -->
        4988 -->
            4832  --> Ausgangsprozess
            6668  -->