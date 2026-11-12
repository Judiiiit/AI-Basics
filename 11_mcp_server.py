#Servidor MCP local (stdio) que publica herramientas de lectura de Weaviate Cloud.

#Se inicia automáticamente desde 09_shopping_list_weaviateMcp.py. No necesita Docker ni un puerto local. 
# Las credenciales se cargan del .env de esta carpeta.


import json
import os
import sys
import unicodedata

import weaviate
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter

load_dotenv()

COLLECTION = "Recetas"
mcp = FastMCP("RecetasWeaviate")


def weaviate_connection():
    url = os.getenv("WEAVIATE_URL", "").strip()
    api_key = os.getenv("WEAVIATE_API_KEY", "").strip()
    if not url or not api_key:
        raise ValueError("Falta WEAVIATE_URL o WEAVIATE_API_KEY en .env")
    return weaviate.connect_to_weaviate_cloud(
        cluster_url=url,
        auth_credentials=Auth.api_key(api_key),
    )


def query_target_vector(collection) -> str | None:
    config = collection.config.get()
    vectors = getattr(config, "vector_config", None)
    if isinstance(vectors, dict) and len(vectors) == 1:
        return next(iter(vectors))
    return None


def normalize_words(value: str) -> str:
    """Normalización básica: minúsculas, tildes y algunos plurales."""
    value = "".join(
        char
        for char in unicodedata.normalize("NFD", value.casefold())
        if unicodedata.category(char) != "Mn"
    )
    words = [word.strip(" .,:;()") for word in value.split()]
    return " ".join(
        word[:-1] if len(word) > 3 and word.endswith("s") else word
        for word in words
    )


def has_required_ingredients(recipe: dict, required: list[str]) -> bool:
    """Exige que TODOS los ingredientes figuren en ingredientesClave."""
    ingredients = [
        normalize_words(x.strip())
        for x in recipe.get("ingredientesClave", "").split(";")
    ]
    for requirement in required:
        target = normalize_words(requirement)
        if target and not any(
            ingredient == target or ingredient.startswith(target + " ")
            for ingredient in ingredients
        ):
            return False
    return True


@mcp.tool()
def check_collection() -> str:
    """Comprueba el acceso de solo lectura a la colección Recetas de Weaviate Cloud."""
    # En stdio, stdout está reservado para mensajes MCP.
    print("[MCP SERVER] check_collection", file=sys.stderr, flush=True)
    with weaviate_connection() as client:
        if not client.collections.exists(COLLECTION):
            raise RuntimeError(f"No existe la colección '{COLLECTION}' en Weaviate")
        collection = client.collections.use(COLLECTION)
        sample = collection.query.fetch_objects(limit=1)
        result = {
            "collection": COLLECTION,
            "connected": True,
            "has_objects": bool(sample.objects),
            "properties": (
                list(sample.objects[0].properties.keys()) if sample.objects else []
            ),
        }
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def search_recipes(query: str, required_ingredients: list[str], max_minutes: int = 0) -> str:
    """Busca recetas por significado; filtra por tiempo y exige todos los ingredientes indicados."""
    print("[MCP SERVER] search_recipes -> Weaviate Cloud", file=sys.stderr, flush=True)
    with weaviate_connection() as client:
        if not client.collections.exists(COLLECTION):
            raise RuntimeError(f"No existe la colección '{COLLECTION}' en Weaviate")
        collection = client.collections.use(COLLECTION)

        filters = (
            Filter.by_property("tiempoMin").less_or_equal(max_minutes)
            if max_minutes > 0
            else None
        )
        kwargs = {"query": query, "limit": 100, "filters": filters}
        target_vector = query_target_vector(collection)
        if target_vector:
            kwargs["target_vector"] = target_vector
        found = collection.query.near_text(**kwargs)

    matches = []
    for obj in found.objects:
        row = obj.properties or {}
        recipe = {
            "idReceta": str(row.get("idReceta") or ""),
            "nombre": str(row.get("nombre") or "Sin nombre"),
            "categoria": str(row.get("categoria") or ""),
            "dieta": str(row.get("dieta") or ""),
            "tiempoMin": row.get("tiempoMin"),
            "raciones": row.get("raciones"),
            "ingredientes": str(row.get("ingredientes") or ""),
            "ingredientesClave": str(row.get("ingredientesClave") or ""),
            "preparacion": str(row.get("preparacion") or ""),
            "descripcion": str(row.get("descripcion") or ""),
        }
        if required_ingredients and not has_required_ingredients(recipe, required_ingredients):
            continue
        matches.append(recipe)
        if len(matches) >= 5:
            break

    # Salida JSON de texto para que sea fácil de inspeccionar en MCP
    return json.dumps(matches, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run(transport="stdio")
