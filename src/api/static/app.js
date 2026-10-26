/**
 * ALPR Vehicle Traceability Dashboard Frontend Logic
 */

document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const searchInput = document.getElementById('search-plate-input');
    const clearBtn = document.getElementById('clear-search-btn');
    const searchBtn = document.getElementById('btn-submit-search');
    const refreshBtn = document.getElementById('btn-refresh');

    const sourceSelect = document.getElementById('filter-source');
    const vehicleInput = document.getElementById('filter-vehicle');
    const timeFromInput = document.getElementById('filter-time-from');
    const timeToInput = document.getElementById('filter-time-to');
    const fuzzyCheck = document.getElementById('check-fuzzy');
    const correctedOnlyCheck = document.getElementById('check-corrected-only');
    const validOnlyCheck = document.getElementById('check-valid-only');
    const unknownOnlyCheck = document.getElementById('check-unknown-only');

    const resultsGrid = document.getElementById('results-grid');

    const resultsCountBadge = document.getElementById('results-count-badge');
    const loadingSpinner = document.getElementById('loading-spinner');
    const emptyState = document.getElementById('empty-state');

    // Stats
    const statTotalSightings = document.getElementById('stat-total-sightings');
    const statTotalValid = document.getElementById('stat-total-valid');
    const statTotalCorrected = document.getElementById('stat-total-corrected');
    const statUniquePlates = document.getElementById('stat-unique-plates');

    // Modal
    const modal = document.getElementById('detail-modal');
    const closeModalBtn = document.getElementById('btn-close-modal');
    const modalPlateTitle = document.getElementById('modal-plate-title');
    const modalBadges = document.getElementById('modal-badges');
    const modalCropImg = document.getElementById('modal-crop-img');
    const trajectoryCanvas = document.getElementById('trajectory-canvas');
    const trajectoryCanvasWrapper = document.getElementById('trajectory-canvas-wrapper');

    const modalSourceName = document.getElementById('modal-source-name');
    const modalVehicleType = document.getElementById('modal-vehicle-type');
    const modalStartTime = document.getElementById('modal-start-time');
    const modalEndTime = document.getElementById('modal-end-time');
    const modalConfValue = document.getElementById('modal-conf-value');
    const modalLockReason = document.getElementById('modal-lock-reason');
    const modalOcrCount = document.getElementById('modal-ocr-count');
    const correctionPanelContent = document.getElementById('correction-panel-content');

    // Initialize Live Stream Link
    const liveStreamLink = document.getElementById('link-live-stream');
    if (liveStreamLink) {
        liveStreamLink.href = `http://${window.location.hostname || 'localhost'}:8080`;
    }

    // Initialize
    loadSources();
    loadStats();
    performSearch();

    // Event Listeners
    searchBtn.addEventListener('click', performSearch);
    searchInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') performSearch();
    });

    clearBtn.addEventListener('click', () => {
        searchInput.value = '';
        searchInput.focus();
        performSearch();
    });

    refreshBtn.addEventListener('click', () => {
        loadStats();
        performSearch();
    });

    // Checkbox auto-search & mutual exclusivity
    validOnlyCheck.addEventListener('change', () => {
        if (validOnlyCheck.checked && unknownOnlyCheck) {
            unknownOnlyCheck.checked = false;
        }
        performSearch();
    });

    if (unknownOnlyCheck) {
        unknownOnlyCheck.addEventListener('change', () => {
            if (unknownOnlyCheck.checked) {
                validOnlyCheck.checked = false;
            }
            performSearch();
        });
    }

    correctedOnlyCheck.addEventListener('change', performSearch);
    fuzzyCheck.addEventListener('change', performSearch);
    sourceSelect.addEventListener('change', performSearch);
    if (vehicleInput) {
        vehicleInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') performSearch();
        });
    }
    timeFromInput.addEventListener('change', performSearch);
    timeToInput.addEventListener('change', performSearch);

    closeModalBtn.addEventListener('click', closeModal);
    modal.addEventListener('click', (e) => {
        if (e.target === modal) closeModal();
    });
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && !modal.classList.contains('hidden')) {
            closeModal();
        }
    });

    // 1. Load Sources into select dropdown
    async function loadSources() {
        try {
            const res = await fetch('/api/sources');
            if (!res.ok) return;
            const sources = await res.json();
            sourceSelect.innerHTML = '<option value="">Tất cả Camera / Video</option>';
            sources.forEach(src => {
                const opt = document.createElement('option');
                opt.value = src.id;
                opt.textContent = `${src.name} (${src.kind.toUpperCase()})`;
                sourceSelect.appendChild(opt);
            });
        } catch (err) {
            console.error('Lỗi nạp danh sách nguồn:', err);
        }
    }

    // 2. Load Stats
    async function loadStats() {
        try {
            const res = await fetch('/api/stats');
            if (!res.ok) return;
            const data = await res.json();
            statTotalSightings.textContent = Number(data.total_sightings || 0).toLocaleString();
            statTotalValid.textContent = Number(data.total_valid || 0).toLocaleString();
            statTotalCorrected.textContent = Number(data.total_corrected || 0).toLocaleString();
            statUniquePlates.textContent = Number(data.total_unique_plates || 0).toLocaleString();
        } catch (err) {
            console.error('Lỗi nạp thống kê:', err);
        }
    }

    // 3. Search Sightings
    async function performSearch() {
        loadingSpinner.classList.remove('hidden');
        emptyState.classList.add('hidden');
        resultsGrid.innerHTML = '';

        const params = new URLSearchParams();
        const plateVal = searchInput.value.trim().toUpperCase();
        if (plateVal) params.append('plate', plateVal);

        params.append('fuzzy', fuzzyCheck.checked ? 'true' : 'false');
        if (sourceSelect.value) params.append('source_id', sourceSelect.value);
        if (vehicleInput && vehicleInput.value.trim()) {
            params.append('vehicle_type', vehicleInput.value.trim());
        }

        if (timeFromInput.value) {
            const tsFrom = new Date(timeFromInput.value).getTime() / 1000;
            params.append('time_from', tsFrom);
        }
        if (timeToInput.value) {
            const tsTo = new Date(timeToInput.value).getTime() / 1000;
            params.append('time_to', tsTo);
        }

        if (correctedOnlyCheck.checked) params.append('corrected_only', 'true');
        if (validOnlyCheck.checked) params.append('valid_only', 'true');
        if (unknownOnlyCheck && unknownOnlyCheck.checked) params.append('unknown_only', 'true');

        console.log('[ALPR Search Request] URL:', `/api/search?${params.toString()}`);

        try {
            const res = await fetch(`/api/search?${params.toString()}`);
            if (!res.ok) throw new Error(`Lỗi API: ${res.status}`);
            const data = await res.json();

            const items = data.results || [];
            resultsCountBadge.textContent = `${items.length} kết quả`;

            if (items.length === 0) {
                emptyState.classList.remove('hidden');
            } else {
                items.forEach(item => {
                    resultsGrid.appendChild(createVehicleCard(item));
                });
            }
        } catch (err) {
            console.error('Lỗi tìm kiếm:', err);
            emptyState.classList.remove('hidden');
            resultsCountBadge.textContent = '0 kết quả';
        } finally {
            loadingSpinner.classList.add('hidden');
        }
    }

    // 4. Create Vehicle Card Element
    function createVehicleCard(item) {
        const card = document.createElement('div');
        card.className = 'vehicle-card';

        const cropUrl = `/api/sightings/${item.id}/crop`;
        const timeFormatted = formatTimestamp(item.t_start);
        const confPercent = Math.round((item.plate_conf || 0) * 100);

        // Badges
        let badgesHtml = '';
        if (item.is_valid) {
            badgesHtml += '<span class="badge badge-valid"><i class="fa-solid fa-check"></i> Chuẩn VN</span>';
        } else {
            badgesHtml += '<span class="badge badge-invalid"><i class="fa-solid fa-triangle-exclamation"></i> UNKNOWN</span>';
        }
        if (item.is_corrected) {
            badgesHtml += '<span class="badge badge-corrected"><i class="fa-solid fa-wand-magic-sparkles"></i> Đã sửa</span>';
        }
        if (item.vehicle_type) {
            badgesHtml += `<span class="badge" style="background: rgba(180, 80, 220, 0.2); color: #d070ff; border: 1px solid rgba(180, 80, 220, 0.4);"><i class="fa-solid fa-motorcycle"></i> ${escapeHtml(item.vehicle_type)}</span>`;
        }
        if (item.fuzzy_score && item.fuzzy_score < 100) {
            badgesHtml += `<span class="badge badge-fuzzy">Khớp ${Math.round(item.fuzzy_score)}%</span>`;
        }

        // Plate text display with before/after highlight
        let plateDisplayHtml = '';
        if (item.is_corrected && item.plate_raw && item.plate_raw !== item.plate) {
            plateDisplayHtml = `
                <div class="plate-text">${escapeHtml(item.plate)}</div>
                <div class="correction-indicator" title="Gốc từ OCR: ${escapeHtml(item.plate_raw)}">
                    <span class="char-badge-from">${escapeHtml(item.plate_raw)}</span>
                    <i class="fa-solid fa-arrow-right"></i>
                    <span class="char-badge-to">${escapeHtml(item.plate)}</span>
                </div>
            `;
        } else {
            plateDisplayHtml = `<div class="plate-text">${escapeHtml(item.plate || 'UNKNOWN')}</div>`;
        }

        card.innerHTML = `
            <div class="card-thumbnail-wrapper">
                <img src="${cropUrl}" alt="Biển số" loading="lazy" onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22100%22 height=%2250%22><rect width=%22100%%22 height=%22100%%22 fill=%22%23222%22/><text x=%2250%%22 y=%2250%%22 dominant-baseline=%22middle%22 text-anchor=%22middle%22 fill=%22%23666%22 font-size=%2212%22>Không có ảnh</text></svg>'">
                <div class="badge-overlay">${badgesHtml}</div>
            </div>
            <div class="card-content">
                <div class="plate-display-box">
                    ${plateDisplayHtml}
                </div>
                <div class="card-meta">
                    <div class="card-meta-row">
                        <span class="meta-item"><i class="fa-solid fa-camera"></i> ${escapeHtml(item.source_name || 'Nguồn')}</span>
                        <span class="meta-item"><i class="fa-solid fa-hashtag"></i> Track #${item.track_id}</span>
                    </div>
                    <div class="card-meta-row">
                        <span class="meta-item"><i class="fa-regular fa-clock"></i> ${timeFormatted}</span>
                        <span>Độ tin cậy: ${confPercent}%</span>
                    </div>
                    <div class="confidence-bar-wrapper">
                        <div class="confidence-bar-fill" style="width: ${confPercent}%"></div>
                    </div>
                </div>
                <button class="btn-view-detail" data-id="${item.id}">
                    <i class="fa-solid fa-expand"></i> Chi Tiết & Quỹ Đạo
                </button>
            </div>
        `;

        card.querySelector('.btn-view-detail').addEventListener('click', () => {
            openDetailModal(item.id);
        });

        return card;
    }

    // 5. Open Detail Modal
    async function openDetailModal(sightingId) {
        try {
            const res = await fetch(`/api/sightings/${sightingId}`);
            if (!res.ok) throw new Error('Không thể tải chi tiết');
            const data = await res.json();

            modalPlateTitle.textContent = data.plate || 'UNKNOWN';

            // Badges
            let bHtml = '';
            if (data.is_valid) {
                bHtml += '<span class="badge badge-valid"><i class="fa-solid fa-check"></i> Biển chuẩn VN</span>';
            } else {
                bHtml += '<span class="badge badge-invalid"><i class="fa-solid fa-triangle-exclamation"></i> Biển UNKNOWN (Sai format / Chưa nhận diện)</span>';
            }
            if (data.is_corrected) {
                bHtml += '<span class="badge badge-corrected"><i class="fa-solid fa-wand-magic-sparkles"></i> Đã hiệu chỉnh ký tự OCR</span>';
            }
            modalBadges.innerHTML = bHtml;


            // Crop image
            modalCropImg.src = `/api/sightings/${data.id}/crop`;

            // Metadata info
            modalSourceName.textContent = `${data.source_name} (${(data.source_kind || '').toUpperCase()})`;
            if (modalVehicleType) {
                modalVehicleType.textContent = data.vehicle_type ? `${data.vehicle_type} (${Math.round((data.vehicle_conf || 0) * 100)}%)` : 'Chưa phân loại';
            }
            modalStartTime.textContent = formatTimestamp(data.t_start);
            modalEndTime.textContent = formatTimestamp(data.t_end);
            modalConfValue.textContent = `${Math.round((data.plate_conf || 0) * 100)}%`;
            modalLockReason.textContent = formatLockReason(data.lock_reason);
            modalOcrCount.textContent = `${data.n_ocr} lần`;

            // Render Corrections Table
            renderCorrectionsPanel(data);

            // Render Trajectory on Context Frame
            renderTrajectory(data);

            modal.classList.remove('hidden');
        } catch (err) {
            console.error('Lỗi hiển thị modal:', err);
            alert('Không thể tải chi tiết lượt xe.');
        }
    }

    function closeModal() {
        modal.classList.add('hidden');
    }

    // 6. Render Corrections Comparison Panel
    function renderCorrectionsPanel(data) {
        if (!data.is_corrected || !data.corrections_detail || data.corrections_detail.length === 0) {
            correctionPanelContent.innerHTML = `
                <p style="color: var(--text-secondary); font-size: 13px;">
                    <i class="fa-solid fa-circle-check" style="color: var(--primary);"></i> 
                    Ký tự nhận diện từ OCR hoàn toàn trùng khớp với định dạng chuẩn, không cần hiệu chỉnh.
                </p>
            `;
            return;
        }

        let rowsHtml = '';
        data.corrections_detail.forEach(c => {
            rowsHtml += `
                <tr>
                    <td>Vị trí #${Number(c.pos) + 1}</td>
                    <td><span class="char-badge-from">${escapeHtml(c.from)}</span></td>
                    <td><span class="char-badge-to">${escapeHtml(c.to)}</span></td>
                    <td style="color: var(--text-muted);">${escapeHtml(c.rule || 'Chuẩn vị trí')}</td>
                </tr>
            `;
        });

        correctionPanelContent.innerHTML = `
            <div style="margin-bottom: 12px; font-size: 13px;">
                Chuỗi OCR thô: <span class="char-badge-from" style="text-decoration:none; background:rgba(239,68,68,0.15); padding:2px 6px; border-radius:4px;">${escapeHtml(data.plate_raw)}</span>
                ➔ Sau hiệu chỉnh: <span class="char-badge-to" style="background:rgba(16,185,129,0.15); padding:2px 6px; border-radius:4px;">${escapeHtml(data.plate)}</span>
            </div>
            <table class="correction-table">
                <thead>
                    <tr>
                        <th>Vị trí</th>
                        <th>Ký tự OCR</th>
                        <th>Sửa thành</th>
                        <th>Lý do</th>
                    </tr>
                </thead>
                <tbody>
                    ${rowsHtml}
                </tbody>
            </table>
        `;
    }

    // 7. Render Trajectory on Canvas
    function renderTrajectory(data) {
        const ctx = trajectoryCanvas.getContext('2d');
        const contextUrl = `/api/sightings/${data.id}/context`;

        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.src = contextUrl;

        img.onload = () => {
            trajectoryCanvas.width = img.width;
            trajectoryCanvas.height = img.height;

            // Draw background frame
            ctx.drawImage(img, 0, 0, img.width, img.height);

            const traj = data.trajectory || [];
            if (traj.length === 0) return;

            // Draw trajectory path
            ctx.lineWidth = 3;
            ctx.strokeStyle = '#06b6d4';
            ctx.shadowColor = '#06b6d4';
            ctx.shadowBlur = 8;
            ctx.beginPath();

            traj.forEach((pt, idx) => {
                const [t, cx, cy] = pt;
                const px = cx * img.width;
                const py = cy * img.height;
                if (idx === 0) {
                    ctx.moveTo(px, py);
                } else {
                    ctx.lineTo(px, py);
                }
            });
            ctx.stroke();
            ctx.shadowBlur = 0; // reset shadow

            // Draw start point
            const [tStart, sx, sy] = traj[0];
            ctx.fillStyle = '#10b981';
            ctx.beginPath();
            ctx.arc(sx * img.width, sy * img.height, 5, 0, Math.PI * 2);
            ctx.fill();

            // Draw end / current vehicle point & bounding box
            const [tEnd, ex, ey, ew, eh] = traj[traj.length - 1];
            ctx.fillStyle = '#f59e0b';
            ctx.beginPath();
            ctx.arc(ex * img.width, ey * img.height, 6, 0, Math.PI * 2);
            ctx.fill();

            // Draw last known bbox if available
            if (ew && eh) {
                const bx = (ex - ew / 2) * img.width;
                const by = (ey - eh / 2) * img.height;
                const bw = ew * img.width;
                const bh = eh * img.height;
                ctx.strokeStyle = '#10b981';
                ctx.lineWidth = 2;
                ctx.strokeRect(bx, by, bw, bh);
            }
        };

        img.onerror = () => {
            // If no context frame available, clear canvas and draw simple trajectory
            trajectoryCanvas.width = 480;
            trajectoryCanvas.height = 270;
            ctx.fillStyle = '#0f172a';
            ctx.fillRect(0, 0, 480, 270);
            ctx.fillStyle = '#64748b';
            ctx.font = '14px Outfit, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('Không có ảnh khung cảnh tương ứng', 240, 135);
        };
    }

    // Helpers
    function formatTimestamp(ts) {
        if (!ts) return '-';
        const d = new Date(ts * 1000);
        return d.toLocaleString('vi-VN', {
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
            day: '2-digit',
            month: '2-digit',
            year: 'numeric',
        });
    }

    function formatLockReason(reason) {
        if (!reason) return 'Chốt tự động';
        if (reason.startsWith('early_lock')) return 'Chốt sớm (Early-lock đồng thuận)';
        if (reason === 'track_end_vote') return 'Bỏ phiếu khi xe rời khung hình';
        return reason;
    }

    function escapeHtml(str) {
        if (!str) return '';
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }
});
