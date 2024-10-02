from crewai_tools import YoutubeVideoSearchTool

# For general search across YouTube content
tool = YoutubeVideoSearchTool()

# custom config
# tool = YoutubeVideoSearchTool(
#     config=dict(
#         llm=dict(provider="ollama", config=dict(model="llama2")),
#         embedder=dict(
#             provider="google",
#             config=dict(model="models/embedding-001", task_type="retrieval_document"),
#         ),
#     )
# )

# For targeted search within a specific video
tool = YoutubeVideoSearchTool(youtube_video_url="https://www.youtube.com/watch?v=Q4RkavtviYU")
