<?php

namespace App\Http\Controllers;

use App\Models\Toilet;
use Illuminate\Http\Request;

class ToiletController extends Controller
{
    /**
     * Display a listing of the resource.
     */
    public function index($floorId)
    {
        return Toilet::where('floor_id', $floorId)->with('gender')->get();
    }

    /**
     * Store a newly created resource in storage.
     */
    public function store(Request $request, $floorId)
    {
        $request->validate([
            'gender_id' => 'required|exists:genders,id',
            'usage_count' => 'nullable|integer',
        ]);

        // Create a new toilet
        $toilet = Toilet::create([
            'floor_id' => $floorId,
            'gender_id' => $request->gender_id,
            'usage_count' => $request->usage_count ?? 0,
        ]);

        return response()->json($toilet, 201);
    }

    /**
     * Display the specified resource.
     */
    public function show($floorId, $toiletId)
    {
        return Toilet::where('floor_id', $floorId)->where('id', $toiletId)->with('gender')->firstOrFail();
    }

    /**
     * Update the specified resource in storage.
     */
    public function update(Request $request, $floorId, $toiletId)
    {
        $toilet = Toilet::where('floor_id', $floorId)->where('id', $toiletId)->firstOrFail();

        $request->validate([
            'gender_id' => 'required|exists:genders,id',
            'usage_count' => 'nullable|integer',
        ]);

        // Update the toilet details
        $toilet->update($request->only('gender_id', 'usage_count'));

        return response()->json($toilet);
    }

    /**
     * Remove the specified resource from storage.
     */
    public function destroy($floorId, $toiletId)
    {
        $toilet = Toilet::where('floor_id', $floorId)->where('id', $toiletId)->firstOrFail();
        $toilet->delete();

        return response()->json(null, 204);
    }

    public function resetUsageCount($toiletId)
    {
        $loggingController = new LoggingCounterController();
        return $loggingController->resetUsageCount(new Request(), $toiletId);
    }
}
