<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class Toilet extends Model
{
    use HasFactory;

    protected $fillable = ['floor_id', 'gender_id', 'usage_count'];

    // Each toilet belongs to a specific floor
    public function floor()
    {
        return $this->belongsTo(Floor::class);
    }

    // Each toilet has a specific gender
    public function gender()
    {
        return $this->belongsTo(Gender::class);
    }

    // Each toilet can have multiple usage logs
    public function usageLogs()
    {
        return $this->hasMany(UsageLog::class);
    }
}
