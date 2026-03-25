import argparse
import asyncio
from pathlib import Path

from app.core.database import init_db_pool, close_db_pool
from app.rag.rag_pipeline import RAGPipeline
from app.agent.report_generator import MedicalReportGenerator
from app.agent.patient_detection import detect_patient
from app.agent.sql_interpreter import SQLInterpreter
from app.agent.sql_tool import run_sql_query
from app.rag.pdf_ingestor import ingest_pdf


async def run_chat(message, patient_id=None):
    rag = RAGPipeline()
    print(await rag.query_async(message, patient_id))


def run_patient_detection(message):
    print(detect_patient(message))


async def run_sql(question):
    interpreter = SQLInterpreter()
    sql_result = await interpreter.interpret(question)
    sql = sql_result.get("sql")
    print("\nSQL:\n", sql)

    if sql:
        params = sql_result.get("params", [])
        params_tuple = tuple(params) if params else None
        result = await run_sql_query(sql, params_tuple)
        print("\nRESULT:\n", result)
    else:
        print("\nERROR:\n", sql_result.get("error"))


async def run_report(patient_id):
    gen = MedicalReportGenerator()
    path = await gen.generate(patient_id)
    print(f"✅ Report generated: {path}")


async def run_ingest(pdf_path, patient_id):
    await ingest_pdf(Path(pdf_path).read_bytes(), patient_id)
    print("✅ PDF ingested")


async def main():
    await init_db_pool()   # 🔑 REQUIRED FOR CLI

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--chat")
        parser.add_argument("--patient-id")
        parser.add_argument("--detect")
        parser.add_argument("--sql")
        parser.add_argument("--report")
        parser.add_argument("--ingest", nargs=2)

        args = parser.parse_args()

        if args.chat:
            await run_chat(args.chat, args.patient_id)
        elif args.detect:
            run_patient_detection(args.detect)
        elif args.sql:
            await run_sql(args.sql)
        elif args.report:
            await run_report(args.report)
        elif args.ingest:
            await run_ingest(*args.ingest)
        else:
            print("No command provided.")

    finally:
        await close_db_pool()  # 🔑 CLEAN SHUTDOWN


if __name__ == "__main__":
    asyncio.run(main())
