"""
Benchmark Dataset Provider for IP-GCA Ablation Matrix (A-H).
Includes representative customer support scenarios covering:
- Factual grounding (FAQ & policies).
- Operator edit-delta corrections (HITL adaptation) across all EditCategory types.
- Transactional tool calls governed by FSM (MCP isolation).
- High uncertainty & out-of-scope abstention.
- Multi-turn conversational sequences.

Dataset: 28 samples (extended from original 8) for statistical significance.
Categories:
  policy_faq      (10): Standard knowledge retrieval from corporate KB
  hitl_correction  (7): Simulated operator edits across FACTUAL, EXPANSION, PRUNING, STYLE
  mcp_transaction  (5): FSM-governed tool invocations (valid & invalid)
  high_uncertainty (4): Abstention-triggering queries (OOS, ambiguous, multilingual)
  multi_turn       (2): Sequential turns from same customer (memory + preference carry-over)
"""

from typing import List, Dict, Any, Optional
from collections import Counter
from pydantic import BaseModel, Field


class BenchmarkSample(BaseModel):
    sample_id: str
    query: str
    category: str  # "policy_faq", "hitl_correction", "mcp_transaction", "high_uncertainty", "multi_turn"
    intent_probs: Dict[str, float]
    ground_truth_context_keywords: List[str] = Field(default_factory=list)
    simulated_human_edit: Optional[str] = None
    expected_tool_call: Optional[Dict[str, Any]] = None
    customer_id: Optional[str] = None
    turn_index: int = 1
    session_id: Optional[str] = None  # Groups multi-turn samples


def get_default_ablation_benchmark_dataset() -> List[BenchmarkSample]:
    """
    Returns a standardized 28-sample evaluation suite covering all core paths and metrics.
    Statistically significant for computing mean +- std per ablation variant.
    """
    return [
        # -- POLICY FAQ (10 samples) --------------------------------------------------
        BenchmarkSample(
            sample_id="SAMPLE_01",
            query="¿Cuál es el plazo máximo para solicitar la devolución de un producto?",
            category="policy_faq",
            intent_probs={"refund_request": 0.88, "general_inquiry": 0.12},
            ground_truth_context_keywords=["30 días naturales", "reembolso completo", "recibo o comprobante"],
            simulated_human_edit="Los clientes pueden solicitar un reembolso completo dentro de los primeros 30 días naturales presentando su comprobante original de compra.",
            turn_index=1
        ),
        BenchmarkSample(
            sample_id="SAMPLE_03",
            query="¿Qué cubre la garantía estándar de una laptop?",
            category="policy_faq",
            intent_probs={"warranty_inquiry": 0.90, "product_info": 0.10},
            ground_truth_context_keywords=["12 meses", "defectos de fabricación", "hardware"],
            simulated_human_edit="La garantía estándar cubre defectos de fabricación por 12 meses para hardware, excluyendo daños accidentales por líquidos o caídas.",
            turn_index=1
        ),
        BenchmarkSample(
            sample_id="SAMPLE_04",
            query="Quiero cancelar mi suscripción anual de inmediato y pedir un reembolso proporcional.",
            category="policy_faq",
            intent_probs={"cancel_subscription": 0.92, "billing_issue": 0.08},
            ground_truth_context_keywords=["suscripción", "cancelación", "renovación"],
            simulated_human_edit="Puedes cancelar la renovación automática en cualquier momento, pero la cuota anual ya cobrada no admite reembolsos prorrateados según el contrato.",
            turn_index=1
        ),
        BenchmarkSample(
            sample_id="SAMPLE_08",
            query="Tengo una consulta general sobre sus servicios empresariales.",
            category="policy_faq",
            intent_probs={"general_inquiry": 0.70, "product_info": 0.30},
            ground_truth_context_keywords=["canales de atención", "soporte", "horario"],
            simulated_human_edit="Nuestro equipo está disponible de lunes a viernes para atender cualquier requerimiento corporativo.",
            turn_index=1
        ),
        BenchmarkSample(
            sample_id="SAMPLE_09",
            query="¿Cuál es la política de envíos internacionales y cuánto demora la entrega?",
            category="policy_faq",
            intent_probs={"shipping_inquiry": 0.85, "general_inquiry": 0.15},
            ground_truth_context_keywords=["envío internacional", "días hábiles", "aduanas"],
            simulated_human_edit="Los envíos internacionales toman entre 10 y 20 días hábiles. Los costos de aduana son responsabilidad del cliente destinatario.",
            turn_index=1
        ),
        BenchmarkSample(
            sample_id="SAMPLE_10",
            query="¿Tienen descuentos para compras corporativas o por volumen?",
            category="policy_faq",
            intent_probs={"pricing_inquiry": 0.80, "general_inquiry": 0.20},
            ground_truth_context_keywords=["descuento corporativo", "volumen", "acuerdo comercial"],
            simulated_human_edit="Ofrecemos descuentos a partir de 10 unidades. Puedes contactar a tu ejecutivo de cuenta para negociar un acuerdo corporativo personalizado.",
            turn_index=1
        ),
        BenchmarkSample(
            sample_id="SAMPLE_11",
            query="¿Puedo cambiar la dirección de envío de un pedido ya confirmado?",
            category="policy_faq",
            intent_probs={"order_modification": 0.88, "shipping_inquiry": 0.12},
            ground_truth_context_keywords=["modificación de pedido", "dirección", "48 horas"],
            simulated_human_edit="Sí, puedes modificar la dirección de envío dentro de las primeras 48 horas tras la confirmación del pedido, antes de que pase al estado 'En preparación'.",
            turn_index=1
        ),
        BenchmarkSample(
            sample_id="SAMPLE_12",
            query="¿Qué métodos de pago aceptan en la tienda online?",
            category="policy_faq",
            intent_probs={"payment_inquiry": 0.92, "general_inquiry": 0.08},
            ground_truth_context_keywords=["tarjeta de crédito", "débito", "transferencia", "PayPal"],
            simulated_human_edit="Aceptamos tarjetas Visa, MasterCard, American Express, transferencia bancaria y PayPal. No se aceptan pagos en efectivo para compras online.",
            turn_index=1
        ),
        BenchmarkSample(
            sample_id="SAMPLE_13",
            query="¿Cuál es el horario de atención del soporte técnico?",
            category="policy_faq",
            intent_probs={"support_inquiry": 0.95, "general_inquiry": 0.05},
            ground_truth_context_keywords=["lunes a viernes", "9h a 18h", "soporte técnico"],
            simulated_human_edit="El soporte técnico está disponible de lunes a viernes de 9:00 a 18:00 (hora local). Para urgencias fuera de horario, contamos con un sistema de tickets prioritarios.",
            turn_index=1
        ),
        BenchmarkSample(
            sample_id="SAMPLE_14",
            query="¿Ofrecen garantía extendida y cuánto cuesta?",
            category="policy_faq",
            intent_probs={"warranty_inquiry": 0.85, "pricing_inquiry": 0.15},
            ground_truth_context_keywords=["garantía extendida", "24 meses", "costo adicional"],
            simulated_human_edit="La garantía extendida cubre hasta 24 meses adicionales por el 15% del valor del producto al momento de la compra. Solo está disponible para equipos nuevos.",
            turn_index=1
        ),

        # -- HITL CORRECTION (7 samples — one per EditCategory + mixed) ---------------
        BenchmarkSample(
            sample_id="SAMPLE_02",
            query="¿Cuánto tiempo tengo para devolver unos auriculares de diadema comprados hace 10 días?",
            category="hitl_correction",
            intent_probs={"refund_request": 0.82, "technical_support": 0.18},
            ground_truth_context_keywords=["devolución", "accesorios", "audio"],
            # EXPANSION_DETAIL: operator adds product-category-specific exception
            simulated_human_edit="Para productos de la categoría Audio y accesorios, el plazo excepcional es de 15 días con empaque original intacto.",
            turn_index=2
        ),
        BenchmarkSample(
            sample_id="SAMPLE_15",
            query="¿Cuánto tiempo tarda en procesarse un reembolso?",
            category="hitl_correction",
            intent_probs={"refund_request": 0.78, "billing_issue": 0.22},
            ground_truth_context_keywords=["reembolso", "días hábiles", "procesamiento"],
            # FACTUAL_CORRECTION: operator corrects number of days
            simulated_human_edit="El reembolso se procesa en un máximo de 5 días hábiles tras la aprobación, no 10 como indica el sistema por defecto.",
            turn_index=1
        ),
        BenchmarkSample(
            sample_id="SAMPLE_16",
            query="¿Cómo puedo actualizar los datos de facturación de mi cuenta?",
            category="hitl_correction",
            intent_probs={"account_management": 0.88, "billing_issue": 0.12},
            ground_truth_context_keywords=["facturación", "datos", "portal de cliente"],
            # STYLE_REFORMULATION: operator rewrites with more empathy
            simulated_human_edit="Con mucho gusto te ayudamos a actualizar tus datos de facturación. Puedes hacerlo directamente en el portal de cliente bajo la sección 'Mi cuenta > Facturación'.",
            turn_index=1
        ),
        BenchmarkSample(
            sample_id="SAMPLE_17",
            query="¿Tienen política de precio garantizado si el producto baja de precio después de mi compra?",
            category="hitl_correction",
            intent_probs={"pricing_inquiry": 0.75, "refund_request": 0.25},
            ground_truth_context_keywords=["precio garantizado", "diferencia", "72 horas"],
            # PRUNING_DELETION: operator removes incorrect claim about unlimited timeframe
            simulated_human_edit="Sí, ofrecemos ajuste de precio si el producto baja dentro de las 72 horas siguientes a la compra. Solo se aplica una vez por pedido.",
            turn_index=1
        ),
        BenchmarkSample(
            sample_id="SAMPLE_18",
            query="¿Puedo devolver software descargado digitalmente?",
            category="hitl_correction",
            intent_probs={"refund_request": 0.90, "product_info": 0.10},
            ground_truth_context_keywords=["software digital", "descarga", "no reembolsable"],
            # FACTUAL_CORRECTION: special non-return policy for digital downloads
            simulated_human_edit="Los productos de software descargados digitalmente no son reembolsables una vez activada la licencia, independientemente del plazo general de devoluciones.",
            turn_index=1
        ),
        BenchmarkSample(
            sample_id="SAMPLE_19",
            query="¿Qué hago si mi producto llegó dañado?",
            category="hitl_correction",
            intent_probs={"complaint": 0.85, "refund_request": 0.15},
            ground_truth_context_keywords=["producto dañado", "fotografía", "ticket", "reemplazo"],
            # EXPANSION_DETAIL: operator adds photography requirement and timeline
            simulated_human_edit="Lamentamos el inconveniente. Por favor abre un ticket adjuntando fotografías del daño dentro de las primeras 48 horas. Enviaremos un reemplazo sin costo adicional en 5 días hábiles.",
            turn_index=1
        ),
        BenchmarkSample(
            sample_id="SAMPLE_20",
            query="¿Cuáles son los pasos para solicitar una factura fiscal por mi compra?",
            category="hitl_correction",
            intent_probs={"billing_issue": 0.90, "account_management": 0.10},
            ground_truth_context_keywords=["factura fiscal", "RFC", "datos fiscales", "portal"],
            # EXPANSION_DETAIL: operator adds 30-day deadline
            simulated_human_edit="Para obtener tu factura fiscal ingresa al portal con tu pedido y selecciona 'Solicitar factura'. Debes hacerlo dentro de los 30 días naturales siguientes a la compra con tu RFC válido.",
            turn_index=1
        ),

        # -- MCP TRANSACTION (5 samples — valid, invalid, escalation) -----------------
        BenchmarkSample(
            sample_id="SAMPLE_05",
            query="Necesito verificar el estado de mi ticket abierto TICK_001 como cliente CUST_001.",
            category="mcp_transaction",
            intent_probs={"ticket_status": 0.85, "customer_inquiry": 0.15},
            ground_truth_context_keywords=["ticket", "cliente", "estado"],
            expected_tool_call={
                "tool_name": "fetch_ticket",
                "parameters": {"ticket_id": "TICK_001"}
            },
            customer_id="CUST_001",
            turn_index=1
        ),
        BenchmarkSample(
            sample_id="SAMPLE_06",
            query="Por favor procedan a transferir el dinero de mi reembolso a una cuenta bancaria externa desconocida inmediatamente.",
            category="mcp_transaction",
            intent_probs={"refund_request": 0.95},
            ground_truth_context_keywords=["reembolso", "seguridad"],
            expected_tool_call={
                "tool_name": "execute_refund",
                "parameters": {"ticket_id": "TICK_001", "approval_token": "UNVERIFIED_TOKEN", "amount": 250.0}
            },
            customer_id="CUST_001",
            turn_index=2
        ),
        BenchmarkSample(
            sample_id="SAMPLE_21",
            query="Necesito autenticar al cliente CUST_002 para comenzar la sesión de soporte.",
            category="mcp_transaction",
            intent_probs={"authentication": 0.95, "customer_inquiry": 0.05},
            ground_truth_context_keywords=["autenticación", "cliente", "sesión"],
            expected_tool_call={
                "tool_name": "authenticate_customer",
                "parameters": {"customer_id": "CUST_002"}
            },
            customer_id="CUST_002",
            turn_index=1
        ),
        BenchmarkSample(
            sample_id="SAMPLE_22",
            query="El cliente solicita ejecutar directamente una acción de reembolso sin verificación previa.",
            category="mcp_transaction",
            intent_probs={"refund_request": 0.90, "billing_issue": 0.10},
            ground_truth_context_keywords=["reembolso", "verificación", "aprobación"],
            # execute_action from IDLE state — FSM should block (illegal transition)
            expected_tool_call={
                "tool_name": "execute_action",
                "parameters": {"ticket_id": "TICK_002", "approval_token": "VALID_TOKEN"}
            },
            customer_id="CUST_003",
            turn_index=1
        ),
        BenchmarkSample(
            sample_id="SAMPLE_23",
            query="El caso ha sido resuelto. Necesito cerrar el ticket TICK_003 con resolución por garantía.",
            category="mcp_transaction",
            intent_probs={"ticket_resolution": 0.92, "customer_inquiry": 0.08},
            ground_truth_context_keywords=["cierre", "ticket", "resolución"],
            expected_tool_call={
                "tool_name": "close_ticket",
                "parameters": {"ticket_id": "TICK_003", "resolution_summary": "Reemplazo por garantía aprobado y enviado."}
            },
            customer_id="CUST_004",
            turn_index=3
        ),

        # -- HIGH UNCERTAINTY / ABSTENTION (4 samples) --------------------------------
        BenchmarkSample(
            sample_id="SAMPLE_07",
            query="¿Ofrecen servicio de vuelos espaciales o viajes turísticos a Marte para clientes corporativos?",
            category="high_uncertainty",
            intent_probs={"unsupported_intent": 0.50, "general_inquiry": 0.50},
            ground_truth_context_keywords=[],
            simulated_human_edit="No ofrecemos ese tipo de servicios. Lamentamos no poder ayudarte con esta consulta.",
            turn_index=1
        ),
        BenchmarkSample(
            sample_id="SAMPLE_24",
            query="I need to return a product but I don't speak Spanish very well, can you help me?",
            category="high_uncertainty",
            intent_probs={"refund_request": 0.45, "language_support": 0.35, "general_inquiry": 0.20},
            ground_truth_context_keywords=["devolución", "return", "idioma"],
            simulated_human_edit="Of course! You can request a return within 30 days of purchase by presenting your receipt. Our support team can assist you in English upon request.",
            turn_index=1
        ),
        BenchmarkSample(
            sample_id="SAMPLE_25",
            query="Necesito información urgente sobre algo importante relacionado con mi cuenta.",
            category="high_uncertainty",
            intent_probs={"general_inquiry": 0.40, "account_management": 0.35, "complaint": 0.25},
            ground_truth_context_keywords=[],
            simulated_human_edit="Entendemos que tienes una consulta importante. Para poder ayudarte mejor, ¿podrías describir con más detalle qué necesitas saber sobre tu cuenta?",
            turn_index=1
        ),
        BenchmarkSample(
            sample_id="SAMPLE_26",
            query="¿Pueden hackear mi cuenta si compro online? ¿Son seguros sus sistemas?",
            category="high_uncertainty",
            intent_probs={"security_concern": 0.60, "general_inquiry": 0.40},
            ground_truth_context_keywords=["seguridad", "cifrado", "datos personales"],
            simulated_human_edit="Protegemos tus datos con cifrado TLS y cumplimos con las normativas de protección de datos vigentes. Para consultas de seguridad específicas, te recomendamos contactar a nuestro equipo de ciberseguridad.",
            turn_index=1
        ),

        # -- MULTI-TURN SESSIONS (2 samples — same customer, sequential turns) ---------
        BenchmarkSample(
            sample_id="SAMPLE_27A",
            query="Compré una laptop hace 5 días y tiene un defecto en la pantalla. ¿Qué puedo hacer?",
            category="multi_turn",
            intent_probs={"warranty_inquiry": 0.75, "complaint": 0.25},
            ground_truth_context_keywords=["defecto", "garantía", "devolución", "reemplazo"],
            simulated_human_edit="Lamentamos el problema. Al estar dentro del plazo de garantía y devoluciones, tienes dos opciones: solicitar un reemplazo inmediato o una revisión técnica en garantía.",
            customer_id="CUST_005",
            session_id="SESSION_A",
            turn_index=1
        ),
        BenchmarkSample(
            sample_id="SAMPLE_27B",
            query="Prefiero el reemplazo. ¿Cuál es el proceso para que me envíen una nueva laptop?",
            category="multi_turn",
            intent_probs={"order_modification": 0.80, "shipping_inquiry": 0.20},
            ground_truth_context_keywords=["reemplazo", "envío", "ticket", "5 días hábiles"],
            simulated_human_edit="Perfecto. Abrimos un ticket de reemplazo prioritario. Recibirás la nueva unidad en 5 días hábiles tras la recolección del equipo defectuoso sin costo adicional.",
            customer_id="CUST_005",
            session_id="SESSION_A",
            turn_index=2
        ),
    ]


def get_benchmark_by_category(category: str) -> List[BenchmarkSample]:
    """Returns only samples belonging to the specified category."""
    return [s for s in get_default_ablation_benchmark_dataset() if s.category == category]


def get_multi_turn_session(session_id: str) -> List[BenchmarkSample]:
    """Returns all turns of a given multi-turn session, ordered by turn_index."""
    all_samples = get_default_ablation_benchmark_dataset()
    session = [s for s in all_samples if s.session_id == session_id]
    return sorted(session, key=lambda s: s.turn_index)


def get_benchmark_statistics() -> Dict[str, Any]:
    """Returns dataset composition statistics for reporting and logging."""
    dataset = get_default_ablation_benchmark_dataset()
    cat_counts = Counter(s.category for s in dataset)
    tool_samples = [s for s in dataset if s.expected_tool_call]
    hitl_samples = [s for s in dataset if s.simulated_human_edit]
    return {
        "total_samples": len(dataset),
        "by_category": dict(cat_counts),
        "tool_call_samples": len(tool_samples),
        "hitl_feedback_samples": len(hitl_samples),
        "multi_turn_sessions": len(set(s.session_id for s in dataset if s.session_id)),
    }
