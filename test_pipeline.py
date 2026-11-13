#!/usr/bin/env python3
import asyncio
import json
import sys
from pathlib import Path

from fastmcp import Client
from video_square_mcp import mcp

CLIP = sys.argv[1]
WORK = Path("/tmp/opencode/vsq_test")
WORK.mkdir(parents=True, exist_ok=True)


async def main():
    async with Client(mcp) as client:
        tools = await client.list_tools()
        print("TOOLS:", [t.name for t in tools])

        async def call(name, args):
            r = await client.call_tool(name, args)
            data = json.loads(r.content[0].text) if r.content else r
            print(
                f"\n== {name}\n{json.dumps(data, indent=1, ensure_ascii=False)[:1500]}"
            )
            return data

        await call("probe", {"clip": CLIP})
        await call("detect_subjects", {"clip": CLIP})
        await call("reframe_square", {"clip": CLIP, "out": str(WORK / "square.mp4")})
        await call(
            "cut",
            {
                "clip": CLIP,
                "ranges": [[1.0, 4.0], [6.0, "00:00:07.5"]],
                "out": str(WORK / "cut.mp4"),
            },
        )
        await call("concat", {"clips": [CLIP, CLIP], "out": str(WORK / "concat.mp4")})
        await call(
            "add_music",
            {
                "video": str(WORK / "square.mp4"),
                "music": str(WORK / "music.mp3"),
                "out": str(WORK / "music_mixed.mp4"),
                "music_db": -18.0,
            },
        )
        await call(
            "export",
            {"video": str(WORK / "square.mp4"), "out": str(WORK / "final.mp4")},
        )
        await call(
            "pipeline",
            {
                "clip": CLIP,
                "out": str(WORK / "pipeline.mp4"),
                "ranges": [[0.5, 7.0]],
                "music": str(WORK / "music.mp3"),
                "music_db": -18.0,
            },
        )


asyncio.run(main())
