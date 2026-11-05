FROM agenthack/base:1.1.1

WORKDIR /app
COPY --chown=agent:agent src/ /app/src/

USER agent
CMD ["python", "-m", "src.server"]
