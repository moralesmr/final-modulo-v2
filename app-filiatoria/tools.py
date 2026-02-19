from langchain_core.tools import tool
from langchain_tavily import TavilySearch


def build_tools(vectorstore):

    @tool
    def buscar_articulo_ccyc(numero: str) -> str:
        """
        Busca un artículo del Código Civil y Comercial por número,
        únicamente dentro del rango 529 a 723 (Derecho de Familia – filiación y temas afines).
        """
        try:
            num = int(numero)
        except ValueError:
            return "Número de artículo inválido."

        if num < 529 or num > 723:
            return "El artículo no pertenece al Derecho de Familia Argentino."

        docs = vectorstore.similarity_search(f"Artículo {num}", k=1)
        if not docs:
            return "Artículo no encontrado en la base documental."

        return docs[0].page_content


    tavily = TavilySearch(max_results=5, topic="general")

    @tool
    def buscar_en_fuentes_juridicas_argentinas(consulta: str) -> str:
        """
        Busca información jurídica argentina en fuentes externas
        (doctrina, jurisprudencia, artículos académicos) si el tema no surge del Código Civil y Comercial.
        """
        query = f"{consulta} derecho argentino"
        res = tavily.run(query)

        if not res or "results" not in res:
            return "No se encontraron fuentes argentinas relevantes."

        return "\n".join(
            f"- {r['title']}: {r['url']}"
            for r in res["results"]
        )

    return [buscar_articulo_ccyc, buscar_en_fuentes_juridicas_argentinas]
