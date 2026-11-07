"""
Unit tests for the IP-GCA Edit-Delta Adaptation Module.
Covers EditDeltaExtractor, PreferenceMemory (temporal decay gamma),
ConditioningBridge, and LoRATriggerController.
"""

import pytest
from pathlib import Path
from src.adaptation.edit_extractor import (
    EditDeltaExtractor,
    EditCategory,
    SemanticDelta
)
from src.adaptation.preference_memory import (
    PreferenceMemory,
    PreferenceKey
)
from src.adaptation.conditioning_bridge import (
    EditDeltaConditioningBridge
)
from src.adaptation.lora_trigger import (
    LoRATriggerController
)


class TestEditDeltaExtractor:
    def test_no_edit_identical_strings(self):
        extractor = EditDeltaExtractor()
        s = "El plazo máximo de devolución es de 15 días hábiles."
        delta = extractor.extract(s, s)

        assert delta.edit_category == EditCategory.NO_EDIT
        assert delta.ksr == 1.0
        assert delta.semantic_similarity == 1.0
        assert not delta.is_significant
        assert len(delta.added_spans) == 0

    def test_factual_correction_numbers(self):
        extractor = EditDeltaExtractor()
        s_gen = "El plazo de garantía es de 6 meses para computadoras."
        s_human = "El plazo de garantía es de 12 meses para computadoras."
        delta = extractor.extract(s_gen, s_human)

        assert delta.edit_category == EditCategory.FACTUAL_CORRECTION
        assert delta.is_significant
        assert delta.ksr < 1.0
        assert "12" in " ".join(delta.added_spans)
        assert "6" in " ".join(delta.removed_spans)

    def test_expansion_detail(self):
        extractor = EditDeltaExtractor()
        s_gen = "Puedes cancelar tu cuenta en ajustes."
        s_human = "Puedes cancelar tu cuenta en ajustes, asegurándote de descargar tus facturas y liquidar saldos pendientes previamente."
        delta = extractor.extract(s_gen, s_human)

        assert delta.edit_category == EditCategory.EXPANSION_DETAIL
        assert delta.is_significant
        assert len(delta.added_spans) > 0

    def test_pruning_deletion(self):
        extractor = EditDeltaExtractor()
        s_gen = "Tu pedido llegará mañana. También te ofrecemos un descuento del 50% en tu próxima compra si calificas a nuestro programa VIP exclusivo."
        s_human = "Tu pedido llegará mañana."
        delta = extractor.extract(s_gen, s_human)

        assert delta.edit_category == EditCategory.PRUNING_DELETION
        assert delta.is_significant
        assert len(delta.removed_spans) > 0


class TestPreferenceMemory:
    def test_add_and_decay_preferences(self):
        memory = PreferenceMemory(min_weight_threshold=0.01)
        extractor = EditDeltaExtractor()

        s_gen = "Reembolso en 5 días."
        s_human = "Reembolso en 10 días hábiles."
        delta = extractor.extract(s_gen, s_human)

        pref = memory.add_preference(
            query="¿Cuándo llega mi reembolso?",
            s_gen=s_gen,
            s_human=s_human,
            delta=delta
        )

        assert pref is not None
        assert pref.current_weight == 1.0
        assert len(memory.keys) == 1

        # Advance 1 turn: weight should be 0.95
        memory.step_turn()
        assert pytest.approx(memory.keys[pref.key_id].current_weight, 0.001) == 0.95

        # Advance 4 more turns (total delta_t = 5): 0.95^5 ~= 0.77378
        for _ in range(4):
            memory.step_turn()
        expected_weight = 0.95 ** 5
        assert pytest.approx(memory.keys[pref.key_id].current_weight, 0.001) == expected_weight

    def test_ignore_insignificant_edits(self):
        memory = PreferenceMemory()
        extractor = EditDeltaExtractor()

        text = "Respuesta exacta sin modificaciones."
        delta = extractor.extract(text, text)

        pref = memory.add_preference("consulta", text, text, delta)
        assert pref is None
        assert len(memory.keys) == 0

    def test_retrieve_relevant(self):
        memory = PreferenceMemory()
        extractor = EditDeltaExtractor()

        d1 = extractor.extract("Plazo 5 días", "Plazo 10 días")
        memory.add_preference("garantía de devolución", "Plazo 5 días", "Plazo 10 días", d1)

        d2 = extractor.extract("Envío gratis", "Envío con costo 15 soles")
        memory.add_preference("costo de envío express", "Envío gratis", "Envío con costo 15 soles", d2)

        results = memory.retrieve_relevant("quiero saber sobre la devolución y garantía", top_k=1)
        assert len(results) == 1
        best_pref, score = results[0]
        assert "garantía" in best_pref.query
        assert score > 0.0


class TestConditioningBridge:
    def test_format_constraints_and_prompt(self):
        bridge = EditDeltaConditioningBridge()
        extractor = EditDeltaExtractor()

        d = extractor.extract("Garantía 6 meses", "Garantía 12 meses extendida")
        pref = PreferenceKey(
            key_id="pref_001",
            query="¿Cuánto dura la garantía?",
            s_gen="Garantía 6 meses",
            s_human="Garantía 12 meses extendida",
            delta=d,
            turn_created=0,
            current_weight=1.0
        )

        constraints_text = bridge.format_preference_constraints([(pref, 0.85)])
        assert "REGLAS DE ALINEACIÓN DEL OPERADOR HUMANO" in constraints_text
        assert "12" in constraints_text
        assert "extendida" in constraints_text
        assert "6" in constraints_text

        prompt = bridge.assemble_generation_prompt(
            user_query="¿Cuál es la garantía del producto?",
            retrieved_passages=["Política de garantías corporativas..."],
            ranked_preferences=[(pref, 0.85)]
        )
        assert "### CONTEXTO DE CONOCIMIENTO EMPRESARIAL" in prompt
        assert "### REGLAS DE ALINEACIÓN DEL OPERADOR HUMANO" in prompt
        assert "### CONSULTA DEL CLIENTE:" in prompt


class TestLoRATriggerController:
    def test_batch_threshold_and_dataset_preparation(self, tmp_path):
        controller = LoRATriggerController()
        # Override threshold to 3 for fast test
        controller.batch_threshold = 3

        memory = PreferenceMemory()
        extractor = EditDeltaExtractor()

        assert not controller.is_threshold_reached(memory)

        for i in range(3):
            d = extractor.extract(f"Sug {i}", f"Hum {i} corregido")
            memory.add_preference(f"Query {i}", f"Sug {i}", f"Hum {i} corregido", d)

        assert controller.is_threshold_reached(memory)

        batch = controller.prepare_training_batch(memory, export_path=str(tmp_path))
        assert batch is not None
        assert batch.total_samples == 3
        assert len(batch.samples) == 3
        assert batch.samples[0]["output"] == "Hum 0 corregido"

        # After processing, unsynced batch should be empty
        assert len(memory.get_unprocessed_batch()) == 0

        # Verify PEFT configuration parameters
        peft_cfg = controller.get_peft_lora_config()
        assert peft_cfg["r"] == 8
        assert peft_cfg["lora_alpha"] == 16
        assert "q_proj" in peft_cfg["target_modules"]

    def test_execute_lora_training_dry_run(self, tmp_path):
        controller = LoRATriggerController()
        controller.batch_threshold = 2
        memory = PreferenceMemory()
        extractor = EditDeltaExtractor()

        for i in range(2):
            d = extractor.extract(f"Sug {i}", f"Hum {i} corregido")
            memory.add_preference(f"Query {i}", f"Sug {i}", f"Hum {i} corregido", d)

        batch = controller.prepare_training_batch(memory)
        assert batch is not None

        out_adapter = tmp_path / "adapters"
        train_result = controller.execute_lora_training(
            batch=batch,
            output_adapter_dir=str(out_adapter),
            dry_run=True
        )

        assert train_result["batch_id"] == batch.batch_id
        assert train_result["sample_count"] == 2
        assert train_result["status"] in ["DRY_RUN_COMPLETED", "SIMULATED_SUCCESS"]
        assert Path(train_result["adapter_path"]).exists()
        assert (Path(train_result["adapter_path"]) / "adapter_config.json").exists()

    def test_evaluate_lora_adaptation_pipeline(self):
        from src.experiments.evaluate_lora_adaptation import run_lora_adaptation_experiment

        summary = run_lora_adaptation_experiment()
        assert "pre_adaptation" in summary
        assert "post_lora_adaptation" in summary
        assert "relative_gains" in summary

        # Alignment and KSR must improve post-adaptation
        assert summary["post_lora_adaptation"]["mean_aop"] > summary["pre_adaptation"]["mean_aop"]
        assert summary["post_lora_adaptation"]["mean_ksr"] > summary["pre_adaptation"]["mean_ksr"]
        assert summary["relative_gains"]["delta_aop"] > 0.30
        assert summary["relative_gains"]["edit_distance_reduction_percent"] > 50.0


