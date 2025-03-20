import importlib.metadata

# 사용자가 제공한 패키지 목록
target_packages = {
    "python-dotenv",
    "pandas",
    "langchain_openai",
    "faiss",
    "numpy",
    "fastapi",
    "jinja2",
    "pydantic",
    "uvicorn",
    "langchain.schema",
    "langchain_community",
    "langchain_core",
    "redis",
    "requests",
    "logging",
    "time",
    "asyncio",
    "concurrent-futures"
}

def generate_filtered_requirements():
    installed_packages = {
        dist.metadata["Name"].lower(): dist.version
        for dist in importlib.metadata.distributions()
    }

    with open("requirements.txt", "w", encoding="utf-8") as file:
        for package in sorted(target_packages):
            package_lower = package.replace("_", "-").replace(".", "-").lower()
            version = installed_packages.get(package_lower)
            if version:
                file.write(f"{package}=={version}\n")
            else:
                print(f"⚠️ {package} 패키지가 설치되지 않았습니다.")

    print("✅ 선택된 패키지들의 requirements.txt 생성 완료!")

generate_filtered_requirements()
