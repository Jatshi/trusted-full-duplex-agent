"""Historical E58 method AST fixture, not a runnable server."""
async def _push_full_duplex(self, payload: Dict[str, Any]) -> None:
    async with self._op_lock:
        await self._wait_finalize()
        input_id = payload.get('input_id')
        audio_base64 = _extract_audio_base64(payload)
        if not audio_base64:
            raise RuntimeError('full_duplex input requires audio')
        audio_waveform = decode_audio_base64(audio_base64)
        decoded_frames = decode_frame_base64_list(_extract_frame_base64_list(payload))
        hints = _first_dict(payload.get('hints'))
        client_force_listen = bool(_coalesce(payload.get('force_listen'), hints.get('force_listen'), default=False))
        mlp_force_listen = False
        requested_client_force_listen = client_force_listen
        client_force_listen = allow_client_cancel(client_force_listen, enabled=self._response_lifecycle_enabled, active=self._active_response_id is not None, controlled_replay=self._controlled_near_end_replay)
        turn_decision = None
        if self._turn_controller is not None:
            if self._active_response_id is not None and client_force_listen:
                self._turn_controller.reset()
                if self._trust_gate is not None:
                    self._trust_gate.reset()
                self._gate_released_voiced_frames = -1
            if self._active_response_id is None or client_force_listen:
                turn_decision = self._turn_controller.feed(audio_waveform)
                self._last_turn_decision = turn_decision
                if self._trust_gate is not None:
                    self._trust_gate.feed(audio_waveform)
            if self._active_response_id is None:
                decision = turn_decision or self._last_turn_decision
                mlp_force_listen = decision is not None and (not decision.allow_speak)
        force_listen = client_force_listen or mlp_force_listen
        gate_decision = None
        gate_intercept = False
        wait_decision = None
        if self._trust_gate is not None and (not force_listen) and (self._active_response_id is None) and (turn_decision is not None) and turn_decision.allow_speak and (turn_decision.voiced_frames > self._gate_released_voiced_frames):
            gate_decision = await asyncio.to_thread(self._trust_gate.evaluate)
            self._last_gate_decision = gate_decision
            gate_intercept = gate_decision.should_intercept
            if gate_decision.allow_execute:
                wait_decision = evaluate_wait_gate(wait_mode=self._wait_mode, transcript=gate_decision.text, trailing_silence_frames=turn_decision.trailing_silence_frames, config=self._wait_gate_config)
                if wait_decision.hold:
                    force_listen = True
                else:
                    self._gate_released_voiced_frames = turn_decision.voiced_frames
            if gate_intercept or gate_decision.listen_only:
                force_listen = True
            logger.info('TFD TrustGate: session=%s action=%s text=%r confidence=%.3f risk=%.3f asr_ms=%.1f', self.session_id, gate_decision.action, gate_decision.text, gate_decision.confidence, gate_decision.risk, gate_decision.asr_ms)
        max_slice_nums = int(_coalesce(payload.get('max_slice_nums'), hints.get('max_slice_nums'), default=1))
        if force_listen:
            logger.info('duplex force-listen: session=%s source=%s client=%s mlp=%s action=%s rms=%s peak=%s threshold=%s', self.session_id, 'tfd_turn_mlp' if mlp_force_listen and (not client_force_listen) else hints.get('source', 'manual_or_unspecified'), client_force_listen, mlp_force_listen, turn_decision.action if turn_decision is not None else None, hints.get('rms'), hints.get('peak'), hints.get('threshold'))
        t0 = time.perf_counter()

        def _duplex_step() -> tuple[Any, float, Dict[str, Any], Dict[str, Any]]:
            prefill_t0 = time.perf_counter()
            prefill_result = self.backend.duplex_prefill(audio_waveform=audio_waveform, frame_list=decoded_frames.frame_list, max_slice_nums=max_slice_nums)
            prefill_ms = (time.perf_counter() - prefill_t0) * 1000
            result = self.backend.duplex_generate(force_listen=force_listen)
            return (result, prefill_ms, prefill_result, self._safe_metrics())
        result, prefill_ms, prefill_result, backend_metrics = await asyncio.to_thread(_duplex_step)
        wall_clock_ms = (time.perf_counter() - t0) * 1000
        metrics = _result_metrics(result, backend_metrics)
        metrics['prefill_ms'] = round(prefill_ms, 1)
        metrics['wall_clock_ms'] = round(wall_clock_ms, 1)
        metrics['tfd_turn_mlp_enabled'] = self._turn_controller is not None
        metrics['tfd_turn_mlp_force_listen'] = mlp_force_listen
        metrics['tfd_client_force_listen'] = client_force_listen
        metrics['tfd_lifecycle_enabled'] = self._response_lifecycle_enabled
        metrics['tfd_unverified_cancel_blocked'] = requested_client_force_listen and (not client_force_listen)
        if turn_decision is not None:
            metrics.update(turn_decision.metrics())
        metrics['tfd_gate_enabled'] = self._trust_gate is not None
        metrics['tfd_gate_evaluated_this_chunk'] = gate_decision is not None
        if gate_decision is not None:
            metrics.update(gate_decision.metrics())
        if wait_decision is not None:
            metrics.update(wait_decision.metrics())
        if isinstance(prefill_result, dict):
            n_vision_slices = prefill_result.get('n_vision_slices', prefill_result.get('n_vision_images'))
            if n_vision_slices is not None:
                metrics['vision_slices'] = n_vision_slices
            prefill_usage = prefill_result.get('usage') or {}
            vision_tokens = prefill_usage.get('input_vision_tokens')
            if vision_tokens is not None:
                metrics['vision_tokens'] = vision_tokens
        chunk_usage = TokenUsage.from_duplex_chunk(prefill_result, result)
        usage_fields = {'chunk_index': self._usage_chunk_index, 'chunk_usage': chunk_usage.to_dict(), 'usage': self._usage.add(chunk_usage)}
        self._usage_chunk_index += 1
        usage_sent = False

        def take_usage_fields() -> Dict[str, Any]:
            nonlocal usage_sent
            if usage_sent:
                return {}
            usage_sent = True
            return usage_fields
        if gate_intercept and gate_decision is not None:
            response_id = str(payload.get('response_id') or f'resp_{uuid.uuid4().hex[:12]}')
            guard_text, guard_audio = _guardrail_response(gate_decision.action)
            await self.send_output_delta('text', session_id=self.session_id, response_id=response_id, input_id=input_id, text=guard_text, reason='guardrail_intercept', metrics=metrics, **take_usage_fields())
            if guard_audio:
                await self.send_output_delta('audio', session_id=self.session_id, response_id=response_id, input_id=input_id, audio=guard_audio, reason='guardrail_intercept', metrics=metrics)
            await self.send_output_delta('listen', session_id=self.session_id, response_id=response_id, input_id=input_id, reason='guardrail_intercept', metrics=metrics)
            if self._turn_controller is not None:
                self._turn_controller.reset()
            self._trust_gate.reset()
            self._gate_released_voiced_frames = -1
            self._schedule_finalize()
            return
        if result.is_listen:
            transition = listen_transition(self._active_response_id, enabled=self._response_lifecycle_enabled, natural_end=result.end_of_turn, client_cancel=client_force_listen, force_listen=force_listen)
            metrics['tfd_response_terminated'] = self._active_response_id is not None and transition['active_response_id'] is None
            await self.send_output_delta('listen', session_id=self.session_id, response_id=self._active_response_id, input_id=input_id, reason=transition['reason'], metrics=metrics, **take_usage_fields())
            self._active_response_id = transition['active_response_id']
            if self._response_lifecycle_enabled and result.end_of_turn and (not client_force_listen):
                if self._turn_controller is not None:
                    self._turn_controller.reset()
                if self._trust_gate is not None:
                    self._trust_gate.reset()
                self._gate_released_voiced_frames = -1
            if gate_decision is not None and gate_decision.listen_only:
                if self._turn_controller is not None:
                    self._turn_controller.reset()
                self._trust_gate.reset()
                self._gate_released_voiced_frames = -1
            self._schedule_finalize()
            return
        if self._active_response_id is None:
            self._active_response_id = str(payload.get('response_id') or f'resp_{uuid.uuid4().hex[:12]}')
        if result.text:
            await self.send_output_delta('text', session_id=self.session_id, response_id=self._active_response_id, input_id=input_id, text=result.text, metrics=metrics, **take_usage_fields())
        if result.audio_data:
            await self.send_output_delta('audio', session_id=self.session_id, response_id=self._active_response_id, input_id=input_id, audio=result.audio_data, metrics=metrics, **take_usage_fields())
        if result.end_of_turn:
            await self.send_output_delta('listen', session_id=self.session_id, response_id=self._active_response_id, input_id=input_id, reason='turn_end', metrics=metrics, **take_usage_fields())
            self._active_response_id = None
            if self._turn_controller is not None:
                self._turn_controller.reset()
            if self._trust_gate is not None:
                self._trust_gate.reset()
            self._gate_released_voiced_frames = -1
        self._schedule_finalize()
