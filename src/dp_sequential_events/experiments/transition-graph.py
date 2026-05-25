import pandas as pd
from graphviz import Digraph
import os

def crear_grafo_transiciones(df, nombre_salida, titulo="Grafo de Transiciones"):
    """
    Genera un Directed-Follows Graph (DFG) con porcentajes y colores.
    """
    # 1. Preparar los datos
    col_time = 'FinalTimestamp' if 'FinalTimestamp' in df.columns else 'Timestamp'
    df[col_time] = pd.to_datetime(df[col_time])
    df = df.sort_values(by=['CaseID', col_time])

    # 2. Calcular frecuencias de los Nodos (Actividades)
    conteo_nodos = df['Activity'].value_counts().to_dict()
    total_eventos = sum(conteo_nodos.values())

    # 3. Calcular frecuencias de las Aristas (Transiciones A -> B)
    # Desplazamos la actividad 1 fila hacia arriba por cada usuario para emparejarlas
    df['Siguiente_Actividad'] = df.groupby('CaseID')['Activity'].shift(-1)
    transiciones = df.dropna(subset=['Siguiente_Actividad'])
    conteo_aristas = transiciones.groupby(['Activity', 'Siguiente_Actividad']).size().to_dict()

    # 4. Configurar Graphviz (Alta calidad para el TFM)
    dot = Digraph(comment=titulo, format='pdf')
    # rankdir='LR' de izquierda a derecha, 'TB' de arriba a abajo.
    # overlap='false' y splines='true' evitan que las flechas pisen los nodos
    dot.attr(rankdir='LR', size='12,12', dpi='300', overlap='false', splines='true')

    # Paleta de colores (Inspirada en tu imagen y en tus eventos)
    colores = {
        'A': '#F4D03F', # Amarillo (Inicio)
        'B': '#5DADE2', # Azul
        'C': '#48C9B0', # Turquesa
        'D': '#F5B041', # Naranja claro
        'E': '#AF7AC5', # Morado
        'F': '#2ECC71', # Verde (Aprobado)
        'G': '#E74C3C', # Rojo (Suspenso)
        'H': '#E67E22', # Naranja oscuro (Abandono)
    }

    # 5. Dibujar Nodos
    for nodo, conteo in conteo_nodos.items():
        nodo_str = str(nodo).upper()
        # Porcentaje del nodo respecto al total de eventos
        pct = (conteo / total_eventos) * 100 
        
        etiqueta = f"{nodo_str}\n{pct:.2f}% | {conteo}"
        color_fondo = colores.get(nodo_str, '#BDC3C7') # Gris por defecto

        dot.node(nodo_str, etiqueta, shape='circle', style='filled',
                 fillcolor=color_fondo, fontcolor='black', 
                 fontname='Helvetica-Bold', width='1.5', fixedsize='true')

    # 6. Dibujar Aristas (Flechas)
    for (origen, destino), conteo in conteo_aristas.items():
        origen_str = str(origen).upper()
        destino_str = str(destino).upper()

        # OPCIONAL: Filtro antimaraña (Spaghetti effect)
        # Si una transición ocurre menos de 10 veces, no la dibujamos para que el grafo se entienda.
        # Puedes cambiar este número o ponerlo a 0 para ver TODAS las flechas.
        if conteo < 5: 
            continue

        # Porcentaje de veces que el nodo origen va a este destino concreto
        total_origen = conteo_nodos[origen]
        pct_salida = (conteo / total_origen) * 100 
        
        etiqueta_flecha = f"{pct_salida:.1f}% | {conteo}"
        
        # La flecha hereda el color del nodo de origen
        color_flecha = colores.get(origen_str, '#BDC3C7')

        dot.edge(origen_str, destino_str, label=etiqueta_flecha, 
                 color=color_flecha, fontcolor=color_flecha, 
                 fontname='Helvetica', fontsize='10', penwidth='1.5')

    # 7. Renderizar y guardar
    dot.render(nombre_salida, cleanup=True)
    print(f"✔ Grafo exportado con éxito: {nombre_salida}.pdf")

# --- BLOQUE DE EJECUCIÓN ---
if __name__ == "__main__":
    from dp_sequential_events.main.main import annotation_and_filtering
    from dp_sequential_events.main.main import sampling_and_anonymization
    
    ruta_dataset = input("Ruta del dataset original: ")
    delta = 0.3
    
    print("Generando grafos...")
    
    # 1. Cargar Original
    df_original = annotation_and_filtering(ruta_dataset, delta, 1, False)
    crear_grafo_transiciones(df_original, "grafo_original")
    
    # 2. Cargar Anonimizado (con test_mode=True para leer las actividades bien)
    df_anon = sampling_and_anonymization(df_original)
    crear_grafo_transiciones(df_anon, "grafo_anonimizado")